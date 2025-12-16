import os
import sys
import json
import pickle
import hashlib
import platform
import tempfile
import traceback
from datetime import datetime
from typing import Optional, Any, Dict

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def _safe_jsonable(obj: Any, max_str_len: int = 20000) -> Any:
    """
    Приводит объект к JSON-совместимому виду максимально безопасно:
    - dict/list/str/int/float/bool/None проходят
    - остальное -> строка (обрезанная)
    """
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_safe_jsonable(x, max_str_len=max_str_len) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[str(k)] = _safe_jsonable(v, max_str_len=max_str_len)
            return out
        # Pandas / numpy
        if hasattr(obj, "tolist"):
            return _safe_jsonable(obj.tolist(), max_str_len=max_str_len)

        s = str(obj)
        if len(s) > max_str_len:
            s = s[:max_str_len] + "...(truncated)"
        return s
    except Exception:
        return "<unserializable>"


def _file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _extract_automl_details(automl: Any) -> Dict[str, Any]:
    """
    Best-effort: пытаемся вытащить "что собрал AutoML" и какие параметры у внутренних моделей.
    LightAutoML не гарантирует стабильные имена внутренних полей, поэтому всё делаем осторожно.
    """
    details: Dict[str, Any] = {
        "automl_class": f"{automl.__class__.__module__}.{automl.__class__.__name__}",
        "automl_str": _safe_jsonable(automl),
        "attrs": {},
        "submodels_with_get_params": [],
    }

    # Список атрибутов, которые часто полезны и встречаются в AutoML-объектах
    candidate_attrs = [
        "task",
        "reader",
        "roles",
        "levels",
        "pipelines",
        "ml_algos",
        "models",
        "blender",
        "timer",
        "params",
        "config",
        "oof_pred",
    ]

    for name in candidate_attrs:
        if hasattr(automl, name):
            try:
                details["attrs"][name] = _safe_jsonable(getattr(automl, name))
            except Exception:
                details["attrs"][name] = "<failed to read>"

    # Дополнительно: попробуем пройтись по некоторым "контейнерным" атрибутам и найти объекты с get_params()
    def collect_get_params(obj: Any, path: str, depth: int = 0, max_depth: int = 3):
        if depth > max_depth:
            return
        try:
            if hasattr(obj, "get_params") and callable(obj.get_params):
                try:
                    params = obj.get_params()
                except Exception:
                    params = "<get_params failed>"
                details["submodels_with_get_params"].append(
                    {
                        "path": path,
                        "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                        "params": _safe_jsonable(params),
                    }
                )
                return

            if isinstance(obj, dict):
                for k, v in obj.items():
                    collect_get_params(v, f"{path}[{k}]", depth + 1, max_depth)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    collect_get_params(v, f"{path}[{i}]", depth + 1, max_depth)
            else:
                # иногда полезно заглянуть во внутренности объектов (но аккуратно)
                if hasattr(obj, "__dict__") and isinstance(obj.__dict__, dict):
                    for k, v in list(obj.__dict__.items())[:50]:
                        collect_get_params(v, f"{path}.{k}", depth + 1, max_depth)

        except Exception:
            return

    for root_name in ["levels", "pipelines", "ml_algos", "models", "blender"]:
        if hasattr(automl, root_name):
            collect_get_params(getattr(automl, root_name), root_name)

    return details


def train_lama_with_mlflow(
    data_path: str,
    target_col: str = "Defect",
    experiment_name: str = "lama_defect_experiment_airflow",
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: str = "/opt/airflow/project/models/lama_model",
) -> Optional[str]:
    """
    Обучает TabularAutoML на data_path, логирует всё в MLflow и сохраняет модель (pickle).
    Пишет в MLflow:
      - params/metrics
      - артефакты: automl.pkl, automl_details.json, data_profile.json, automl_summary.txt, error.txt (если был)
    Возвращает путь к файлу модели (pickle).
    Если случилась ошибка — пробрасывает исключение (чтобы Airflow тоже пометил задачу как Failed).
    """
    # 1) MLflow tracking (контейнерный вариант по умолчанию)
    default_mlruns_dir = "/opt/airflow/mlruns"
    os.makedirs(default_mlruns_dir, exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file:{default_mlruns_dir}")
    mlflow.set_tracking_uri(tracking_uri)

    # 2) Данные
    df = pd.read_csv(data_path)

    # Берём каждую 100-ю строку: 0, 100, 200, ...
    step = 60
    df = df.iloc[::step].reset_index(drop=True)

    # Простая “профилизация” данных для артефакта
    data_profile = {
        "data_path": data_path,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "target_col": target_col,
        "target_value_counts": _safe_jsonable(df[target_col].value_counts(dropna=False).to_dict())
        if target_col in df.columns else None,
        "na_count_total": int(df.isna().sum().sum()),
        "na_count_by_col": _safe_jsonable(df.isna().sum().to_dict()),
    }

    # Доп. метаданные про файл (полезно для воспроизводимости)
    try:
        data_profile["file_size_bytes"] = int(os.path.getsize(data_path))
        data_profile["file_mtime"] = datetime.fromtimestamp(os.path.getmtime(data_path)).isoformat()
        data_profile["file_md5"] = _file_md5(data_path)
    except Exception:
        data_profile["file_size_bytes"] = None
        data_profile["file_mtime"] = None
        data_profile["file_md5"] = None

    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col] if target_col in df.columns else None,
    )

    roles = {"target": target_col}
    task = Task("binary")

    # 3) Эксперимент
    mlflow.set_experiment(experiment_name)

    # Артефакты — складываем во временную папку, потом одним вызовом логируем в MLflow
    with tempfile.TemporaryDirectory() as tmpdir:
        # Базовая инфа об окружении (как артефакт)
        env_info = {
            "python": sys.version,
            "platform": platform.platform(),
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
        }
        env_info_path = os.path.join(tmpdir, "env_info.json")
        with open(env_info_path, "w", encoding="utf-8") as f:
            json.dump(env_info, f, ensure_ascii=False, indent=2)

        data_profile_path = os.path.join(tmpdir, "data_profile.json")
        with open(data_profile_path, "w", encoding="utf-8") as f:
            json.dump(data_profile, f, ensure_ascii=False, indent=2)

        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id

                # Params (видно в UI)
                mlflow.log_param("data_path", data_path)
                mlflow.log_param("target_col", target_col)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("tracking_uri", tracking_uri)
                mlflow.log_param("experiment_name", experiment_name)
                mlflow.log_param("run_id", run_id)

                # 4) Обучение
                automl = TabularAutoML(task=task)
                _ = automl.fit_predict(train, roles=roles, valid_data=valid)

                # 5) Метрика
                valid_pred = automl.predict(valid)
                y_true = valid[target_col].values
                y_score = valid_pred.data[:, 0]
                auc = roc_auc_score(y_true, y_score)

                print(f"[LAMA] Validation ROC-AUC: {auc:.6f}")
                mlflow.log_metric("valid_roc_auc", float(auc))

                # 6) Сохраняем модель (pickle) в уникальную папку по run_id
                run_model_dir = os.path.join(model_dir, run_id)
                os.makedirs(run_model_dir, exist_ok=True)
                model_path = os.path.join(run_model_dir, "automl.pkl")

                with open(model_path, "wb") as f:
                    pickle.dump(automl, f)

                # 7) “Что выбрал AutoML” + параметры внутренних моделей (best-effort)
                automl_details = _extract_automl_details(automl)

                automl_details_path = os.path.join(tmpdir, "automl_details.json")
                with open(automl_details_path, "w", encoding="utf-8") as f:
                    json.dump(automl_details, f, ensure_ascii=False, indent=2)

                automl_summary_path = os.path.join(tmpdir, "automl_summary.txt")
                with open(automl_summary_path, "w", encoding="utf-8") as f:
                    f.write(str(automl) + "\n")

                # 8) Логирование артефактов в MLflow
                # Модель отдельным артефактом (чтобы точно была вкладка Artifacts -> model/)
                mlflow.log_artifact(model_path, artifact_path="model")

                # Остальные файлы (профиль данных, детали AutoML, окружение)
                mlflow.log_artifacts(tmpdir, artifact_path="reports")

                return model_path

        except Exception as e:
            # Пишем причину ошибки как артефакт, чтобы в MLflow было понятно "почему Failed"
            err_path = os.path.join(tmpdir, "error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write("Exception:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())

            # Пытаемся залогировать ошибку, если run активен (best-effort)
            try:
                if mlflow.active_run() is not None:
                    mlflow.log_artifacts(tmpdir, artifact_path="reports")
            except Exception:
                pass

            print(f"[LAMA] Training failed: {e}")
            # Важно: пробрасываем ошибку наверх, чтобы Airflow пометил задачу как Failed
            raise


if __name__ == "__main__":
    DATA_PATH = "/opt/airflow/project/train_with_mean_amplitude_and_defect/U_PD4_with_defect_mean_amplitude_defect.csv"
    model_path = train_lama_with_mlflow(DATA_PATH)
    print(f"Модель сохранена в: {model_path}")
