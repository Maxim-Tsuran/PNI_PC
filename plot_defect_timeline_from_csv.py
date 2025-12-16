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
    df = pd.read_csv(csv_path)

    required = ['Time', 'Predicted_Defect', 'Defect_Probability']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"В файле {csv_path} отсутствует обязательный столбец '{col}'")

    has_defect = 'Defect' in df.columns

    roc_auc = None
    acc = None
    if has_defect:
        fpr, tpr, _ = roc_curve(df['Defect'], df['Defect_Probability'])
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(df['Defect'], df['Predicted_Defect'])

    real_defect_time = None
    if has_defect:
        real_idx = df.index[df['Defect'] == 1]
        if len(real_idx) > 0:
            real_defect_time = df.loc[real_defect_time := real_idx[0], 'Time']  # noqa

    pred_defect_time = find_predicted_defect_time(
        predicted_defect=df['Predicted_Defect'].values,
        time=df['Time'].values,
        min_consecutive=min_consecutive
    )

    time_error = None
    time_error_pct = None
    if (real_defect_time is not None) and (pred_defect_time is not None):
        time_error = pred_defect_time - real_defect_time
        if real_defect_time != 0:
            time_error_pct = (time_error / real_defect_time) * 100.0

    file_name = os.path.basename(csv_path)

    # 1) БОЛЬШЕ ХОЛСТ И ПОЛЯ
    fig, ax = plt.subplots(figsize=(18, 5.5))
    plt.subplots_adjust(left=0.08, right=0.8, top=0.88, bottom=0.2)  # место для подписей и вынесенной легенды

    # 2) ЛИНИИ
    if has_defect:
        ax.plot(df['Time'], df['Defect'], label='Реальный дефект',
                drawstyle='steps-post', color='green', linewidth=2)
    ax.plot(df['Time'], df['Predicted_Defect'], label='Предсказанный дефект',
            drawstyle='steps-post', color='red', alpha=0.85, linewidth=2)



    # 3) МОМЕНТЫ
    if real_defect_time is not None:
        ax.axvline(real_defect_time, color='green', linestyle='--', linewidth=2, label='Момент Реального дефекта')
    if pred_defect_time is not None:
        ax.axvline(pred_defect_time, color='red', linestyle='--', linewidth=2, label='Момент предсказания')

    # 4) ГРАНИЦЫ И ТИКИ
    ax.set_xlim(df['Time'].min(), df['Time'].max())
    ax.set_ylim(-0.1, 1.65)  # выше, чтобы рамка не резала аннотации
    ax.tick_params(axis='both', labelsize=12)

    # 5) АННОТАЦИИ — ОТДВИГАЕМ ДАЛЬШЕ ОТ ЛИНИЙ
    if real_defect_time is not None:
        ax.annotate(f'Реальный дефект\n{real_defect_time:.2f} c',
                    xy=(real_defect_time, 1), xycoords='data',
                    xytext=(-30, 42), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-|>', color='green', lw=2, shrinkA=0, shrinkB=0, relpos=(0, 0.5)),
                    fontsize=20, color='green', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.9))

    if pred_defect_time is not None:
        ax.annotate(f'Предсказанный дефект\n{pred_defect_time:.2f} c',
                    xy=(pred_defect_time, 1), xycoords='data',
                    xytext=(48, 42), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2, shrinkA=0, shrinkB=0, relpos=(1, 0.5)),
                    fontsize=20, color='red', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.9))
    else:
        ax.text(df['Time'].iloc[-1], 1.10, 'Дефект не обнаружен моделью',
                color='red', fontsize=20, va='bottom', ha='right')

    # 6) ПОДПИСИ ОСЕЙ И ЗАГОЛОВОК
    ax.set_title(f"{file_name}", fontsize=16)
    ax.set_xlabel('Время', fontsize=20)
    ax.set_ylabel('Метка дефекта', fontsize=20)

    # 7) ЛЕГЕНДА — ВНЕ ГРАФИКА СПРАВА
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0., prop={'size': 20}, framealpha=0.9)

    # 8) ВРЕЗКА С МЕТРИКАМИ — НИЗ ПРАВО
    meta_lines = []
    if has_defect and roc_auc is not None:
        meta_lines.append(f"AUC: {roc_auc:.3f}")
    if has_defect and acc is not None:
        meta_lines.append(f"Accuracy: {acc:.3f}")
    info_text = "\n".join(meta_lines) if meta_lines else " "

    at = AnchoredText(info_text, loc='lower right',
                      prop=dict(size=13, color='black'),
                      frameon=True, pad=0.3, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor('0.35')
    ax.add_artist(at)
    # Чуть сместим врезку внутрь
    at.set_bbox_to_anchor((0.975, 0.03), transform=ax.transAxes)

    # 9) КРАСНЫЙ БЛОК «ОШИБКА» — ВНИЗУ СПРАВА, НЕ ПЕРЕКРЫВАЕТ
    if time_error is not None:
        err_text = f"Ошибка: {time_error:.2f} c" + (f"  ({time_error_pct:.2f}%)" if time_error_pct is not None else "")
        fig.text(0.945, 0.18, err_text,  # подберите x/y под свой макет
                 ha='right', va='bottom',
                 fontsize=20, color='crimson', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='crimson', alpha=0.95))

    # 10) ФИНАЛЬНАЯ ПОДГОТОВКА
    fig.tight_layout()

    # Сохранение
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]
    if base_name.endswith('_with_defect'):
        base_name = base_name.replace('_with_defect', '')
    out_path = os.path.join(output_folder, f'comparre_defect_{base_name}.png')
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
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
