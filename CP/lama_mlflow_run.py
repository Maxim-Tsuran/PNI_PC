import os
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML

os.makedirs("/opt/airflow/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:/opt/airflow/mlruns")

DATA_PATH = "/opt/airflow/project/train_with_mean_amplitude_and_defect/U_PD4_with_defect_mean_amplitude_defect.csv"
TARGET_COL = "Defect"


def main():
    # 1. Загружаем данные
    df = pd.read_csv(DATA_PATH)

    train, valid = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    roles = {"target": TARGET_COL}
    task = Task("binary")

    # 2. Настройка эксперимента MLflow
    mlflow.set_experiment("lama_defect_experiment")

    with mlflow.start_run():
        # Логируем базовую информацию о данных
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("target_col", TARGET_COL)

        # 3. Обучаем LightAutoML
        automl = TabularAutoML(task=task)
        oof = automl.fit_predict(train, roles=roles, valid_data=valid)

        # 4. Считаем ROC-AUC на валидации
        valid_pred = automl.predict(valid)
        y_true = valid[TARGET_COL].values
        y_score = valid_pred.data[:, 0]

        auc = roc_auc_score(y_true, y_score)
        print(f"Validation ROC-AUC: {auc:.4f}")

        # Логируем метрику в MLflow
        mlflow.log_metric("valid_roc_auc", auc)

        # 5. Сохраняем модель как артефакт (через pickle)
        import pickle

        model_dir = "lama_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "automl.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(automl, f)

        mlflow.log_artifact(model_path, artifact_path="model")


if __name__ == "__main__":
    main()
