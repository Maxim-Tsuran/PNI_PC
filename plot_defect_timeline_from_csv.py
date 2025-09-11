# plot_defect_timeline_from_csv.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from matplotlib.offsetbox import AnchoredText

def find_predicted_defect_time(predicted_defect: np.ndarray, time: np.ndarray, min_consecutive: int) -> float | None:
    """Поиск времени начала дефекта по первой серии из min_consecutive подряд идущих 1 в массиве предсказаний."""
    count = 0
    for i in range(len(predicted_defect)):
        if predicted_defect[i] == 1:
            count += 1
            if count == min_consecutive:
                return time[i - min_consecutive + 1]
        else:
            count = 0
    return None

def plot_from_predicted_csv(csv_path: str, output_folder: str, min_consecutive: int = 40) -> str:
    """
    Строит график сравнения меток дефекта по уже рассчитанному predicted CSV:
    - 'Реальный дефект' (истинный Defect, если есть)
    - 'Предсказанный дефект' (Predicted_Defect)
    - Вертикальные линии реального и предсказанного времени дефекта
    - Врезка внизу справа с AUC/Accuracy, а также красный блок с Ошибкой (сек и %)
    Данные не пересчитывает, не изменяет.
    """
    df = pd.read_csv(csv_path)

    # Проверка обязательных столбцов
    required = ['Time', 'Predicted_Defect', 'Defect_Probability']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"В файле {csv_path} отсутствует обязательный столбец '{col}'")

    has_defect = 'Defect' in df.columns

    # ROC/AUC и Accuracy при наличии истинных меток
    roc_auc = None
    acc = None
    if has_defect:
        fpr, tpr, _ = roc_curve(df['Defect'], df['Defect_Probability'])
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(df['Defect'], df['Predicted_Defect'])

    # Время реального дефекта (если есть)
    real_defect_time = None
    if has_defect:
        real_idx = df.index[df['Defect'] == 1]
        if len(real_idx) > 0:
            real_defect_time = df.loc[real_idx[0], 'Time']

    # Время предсказанного дефекта по уставке
    pred_defect_time = find_predicted_defect_time(
        predicted_defect=df['Predicted_Defect'].values,
        time=df['Time'].values,
        min_consecutive=min_consecutive
    )

    # Ошибка времени (сек) и (%) при наличии real_defect_time
    time_error = None
    time_error_pct = None
    if (real_defect_time is not None) and (pred_defect_time is not None):
        time_error = pred_defect_time - real_defect_time
        if real_defect_time != 0:
            time_error_pct = (time_error / real_defect_time) * 100.0

    # Построение
    file_name = os.path.basename(csv_path)
    fig, ax = plt.subplots(figsize=(14, 4))

    # Ступенчатые линии
    if has_defect:
        ax.plot(df['Time'], df['Defect'], label='Реальный дефект',
                drawstyle='steps-post', color='green', linewidth=2)
    ax.plot(df['Time'], df['Predicted_Defect'], label='Предсказанный дефект',
            drawstyle='steps-post', color='red', alpha=0.7, linewidth=2)

    # Вертикальные отметки
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
        ax.text(df['Time'].iloc[-1], 1.10, 'Дефект не обнаружен моделью',
                color='red', fontsize=12, va='bottom', ha='right')

    # Оси, легенда, заголовок
    ax.set_xlim(df['Time'].min(), df['Time'].max())
    ax.set_title(f"{file_name}", fontsize=14)
    ax.set_xlabel('Время')
    ax.set_ylabel('Метка дефекта')
    ax.legend(loc='upper right')

    # Врезка с метриками (AUC/Accuracy) — чёрный блок снизу справа
    meta_lines = []
    if has_defect and roc_auc is not None:
        meta_lines.append(f"AUC: {roc_auc:.3f}")
    if has_defect and acc is not None:
        meta_lines.append(f"Accuracy: {acc:.3f}")
    info_text = "\n".join(meta_lines) if meta_lines else " "

    at = AnchoredText(info_text, loc='lower right', prop=dict(size=11, color='black'),
                      frameon=True, pad=0.3, borderpad=3.0)
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor('0.35')
    ax.add_artist(at)

    # Красный блок ошибки: секунды + проценты (если возможно)
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
    if base_name.endswith('_with_defect'):
        base_name = base_name.replace('_with_defect', '')
    out_path = os.path.join(output_folder, f'comparre_defect_{base_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"График сравнения меток дефекта сохранён: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Построение графиков сравнения дефектов по predicted CSV без пересчёта данных/модели."
    )
    parser.add_argument('--src', type=str, default='predicted_row', help='Папка с *_predicted.csv')
    parser.add_argument('--dst', type=str, default='plots_from_predicted', help='Папка для сохранения графиков')
    parser.add_argument('--min_consecutive', type=int, default=40, help='Уставка: подряд идущие 1 для фиксации дефекта')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        if name.endswith("_predicted.csv"):
            plot_from_predicted_csv(os.path.join(args.src, name), args.dst, min_consecutive=args.min_consecutive)

if __name__ == "__main__":
    main()
