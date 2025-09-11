# plot_defect_timeline_from_lr_csv.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from matplotlib.offsetbox import AnchoredText

# Конфигурация, синхронная вашему пайплайну
INPUT_FOLDER = "test_with_mean_amplitude_and_defect"
PRED_FOLDER = "linear_regression_catboost_predictions"
OUTPUT_FOLDER = "plots_from_lr_predicted"
TRAIN_SIZE = 150000  # должно совпадать с predict_linear_regression_catboost.py

def find_predicted_defect_time(predicted_defect: np.ndarray, time: np.ndarray, min_consecutive: int) -> float | None:
    """Поиск времени начала дефекта по первой серии из min_consecutive подряд идущих 1."""
    count = 0
    for i in range(len(predicted_defect)):
        if predicted_defect[i] == 1:
            count += 1
            if count == min_consecutive:
                return time[i - min_consecutive + 1]
        else:
            count = 0
    return None

def load_real_defect_and_time(raw_csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    ВОССТАНАВЛИВАЕТ real_defect и full_time ТОЧНО КАК В РАБОЧЕМ СКРИПТЕ:
    real_defect = concat(train_df['Defect'], test_df['Defect'])
    full_time   = concat(train_df['Time'],   test_df['Time'])
    """
    src = pd.read_csv(raw_csv_path)

    if 'Defect' not in src.columns or 'Time' not in src.columns:
        raise ValueError(f"В исходном файле нет столбцов Defect/Time: {raw_csv_path}")

    # Разбиение на train/test в том же месте:
    if len(src) <= TRAIN_SIZE:
        raise ValueError(f"Исходный файл короче или равен TRAIN_SIZE: {raw_csv_path}")

    train_df = src.iloc[:TRAIN_SIZE]
    test_df = src.iloc[TRAIN_SIZE:]

    real_defect = np.concatenate([train_df['Defect'].values, test_df['Defect'].values])
    full_time = np.concatenate([train_df['Time'].values, test_df['Time'].values])
    return real_defect, full_time

def plot_from_lr_predicted_csv(pred_csv_path: str, raw_csv_path: str, output_folder: str,
                               min_consecutive: int = 40) -> str:
    """
    Строит график сравнения меток дефекта, читая:
      - predicted CSV: Time, Predicted_Defect, Defect_Probability
      - исходный RAW CSV из INPUT_FOLDER: Defect, Time
    Ничего не пересчитывает.
    """
    df_pred = pd.read_csv(pred_csv_path)

    # Проверяем наличие нужных столбцов в predicted
    for col in ['Time', 'Predicted_Defect', 'Defect_Probability']:
        if col not in df_pred.columns:
            raise ValueError(f"В predicted CSV отсутствует столбец '{col}': {pred_csv_path}")

    # Восстанавливаем real_defect и full_time из исходного raw файла
    real_defect, full_time = load_real_defect_and_time(raw_csv_path)

    # Из predicted берём предсказанную ступень и вероятности
    pred = df_pred['Predicted_Defect'].astype(int).values
    proba = df_pred['Defect_Probability'].astype(float).values

    # Контроль синхронизации длин (должны совпасть)
    if len(full_time) != len(pred):
        raise ValueError(f"Длины full_time({len(full_time)}) и Predicted_Defect({len(pred)}) не совпадают "
                         f"для пары файлов:\nRAW: {raw_csv_path}\nPRED: {pred_csv_path}")

    # Время реального дефекта (первая единица)
    real_defect_idx = np.where(real_defect == 1)[0]
    real_defect_time = full_time[real_defect_idx[0]] if len(real_defect_idx) > 0 else None

    # Время предсказанного дефекта по уставке
    pred_defect_time = find_predicted_defect_time(pred, full_time, min_consecutive)

    # Ошибка времени (сек) и в процентах
    time_error = None
    time_error_pct = None
    if (real_defect_time is not None) and (pred_defect_time is not None):
        time_error = pred_defect_time - real_defect_time
        if real_defect_time != 0:
            time_error_pct = (time_error / real_defect_time) * 100.0

    # Метрики AUC/Accuracy по всей последовательности
    roc_auc = None
    acc = None
    if len(np.unique(real_defect)) > 1:  # чтобы ROC был корректным
        fpr, tpr, _ = roc_curve(real_defect, proba)
        roc_auc = auc(fpr, tpr)
    acc = accuracy_score(real_defect, pred)

    # Построение
    file_name = os.path.basename(pred_csv_path)
    fig, ax = plt.subplots(figsize=(14, 4))

    # Зелёная ступень (истинный дефект)
    ax.plot(full_time, real_defect, label='Истинный дефект',
            drawstyle='steps-post', color='green', linewidth=2)

    # Красная ступень (предсказанный дефект)
    ax.plot(full_time, pred, label='Предсказанный дефект',
            drawstyle='steps-post', color='red', alpha=0.85, linewidth=2)

    # Моменты (штриховые)
    if real_defect_time is not None:
        ax.axvline(real_defect_time, color='green', linestyle='--', linewidth=2, label='Момент Реального дефекта')
    if pred_defect_time is not None:
        ax.axvline(pred_defect_time, color='red', linestyle='--', linewidth=2, label='Момент предсказания')

    # Поднять верхнюю границу
    ax.set_ylim(-0.1, 1.54)

    # Аннотации стрелками
    if real_defect_time is not None:
        ax.annotate(f'Реальный дефект\n{real_defect_time:.2f} c',
                    xy=(real_defect_time, 1), xycoords='data',
                    xytext=(-70, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="-|>", color='green', lw=1.5, shrinkA=0, shrinkB=0),
                    fontsize=11, color='green', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='green', alpha=0.85))
    if pred_defect_time is not None:
        ax.annotate(f'Предсказанный дефект\n{pred_defect_time:.2f} c',
                    xy=(pred_defect_time, 1), xycoords='data',
                    xytext=(20, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="-|>", color='red', lw=1.5, shrinkA=0, shrinkB=0),
                    fontsize=11, color='red', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='red', alpha=0.85))
    else:
        ax.text(full_time[-1], 1.10, 'Дефект не обнаружен моделью',
                color='red', fontsize=12, va='bottom', ha='right')

    # Оси/легенда/заголовок
    ax.set_xlim(full_time.min(), full_time.max())
    ax.set_title(f"{file_name}", fontsize=14)
    ax.set_xlabel('Время')
    ax.set_ylabel('Метка дефекта')
    ax.legend(loc='upper right')

    # Врезка с AUC/Accuracy
    meta_lines = []
    if roc_auc is not None:
        meta_lines.append(f"AUC: {roc_auc:.3f}")
    if acc is not None:
        meta_lines.append(f"Accuracy: {acc:.3f}")
    info_text = "\n".join(meta_lines) if meta_lines else " "

    at = AnchoredText(info_text, loc='lower right', prop=dict(size=11, color='black'),
                      frameon=True, pad=0.3, borderpad=3.0)
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor('0.35')
    ax.add_artist(at)

    # Красный блок Ошибки (сек + %)
    if time_error is not None:
        if time_error_pct is not None:
            err_text = f"Ошибка: {time_error:.2f} c  ({time_error_pct:.2f}%)"
        else:
            err_text = f"Ошибка: {time_error:.2f} c"
        ax.text(0.99, 0.04, err_text,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, color='crimson', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='crimson', alpha=0.95))

    fig.tight_layout()

    # Сохранение
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]
    if base_name.startswith("predicted_"):
        base_name = base_name[len("predicted_"):]
    out_path = os.path.join(output_folder, f'compare_defect_{base_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"График сравнения меток дефекта сохранён: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(
        description="Построение графиков сравнения дефектов по predicted CSV (LR-пайплайн) без пересчёта."
    )
    parser.add_argument('--raw', type=str, default=INPUT_FOLDER,
                        help='Папка с исходными CSV (Time, Defect и др.)')
    parser.add_argument('--pred', type=str, default=PRED_FOLDER,
                        help='Папка с predicted_*.csv, созданными пайплайном')
    parser.add_argument('--dst', type=str, default=OUTPUT_FOLDER,
                        help='Папка для сохранения графиков')
    parser.add_argument('--min_consecutive', type=int, default=40,
                        help='Уставка: подряд идущие 1 для фиксации дефекта')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Идём по всем predicted_*.csv, ищем исходный raw CSV с тем же базовым именем
    for name in os.listdir(args.pred):
        if not (name.endswith(".csv") and name.startswith("predicted_")):
            continue
        pred_path = os.path.join(args.pred, name)
        base = name[len("predicted_"):]  # исходное имя файла
        raw_path = os.path.join(args.raw, base)
        if not os.path.exists(raw_path):
            print(f"Пропущено: нет исходного файла для {pred_path} -> ожидается {raw_path}")
            continue
        try:
            plot_from_lr_predicted_csv(pred_path, raw_path, args.dst, min_consecutive=args.min_consecutive)
        except Exception as e:
            print(f"Ошибка при построении для {name}: {e}")

if __name__ == "__main__":
    main()
