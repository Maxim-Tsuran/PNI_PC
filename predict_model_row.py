import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc

# === 1. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
model = joblib.load('catboost_defect_predictor.cbm')

# === 2. ПОИСК CSV-ФАЙЛОВ ДЛЯ ПРЕДСКАЗАНИЯ ===
predict_files = glob.glob("test_with_defect/*.csv")
output_folder = "predicted_row"
os.makedirs(output_folder, exist_ok=True)

# === ДЛЯ ОБЩЕЙ ROC-КРИВОЙ ===
all_y_true = []
all_y_proba = []

# === ДЛЯ ОЦЕНКИ ТОЧНОСТИ ВРЕМЕНИ ДЕФЕКТА ===
min_consecutive = 40  # Уставка: минимальное количество подряд Predicted_Defect=1
time_errors = []

# === 3. ОБРАБОТКА КАЖДОГО ФАЙЛА ===
for idx, file in enumerate(predict_files):
    df = pd.read_csv(file)

    if not {'Time', 'PD_Level'}.issubset(df.columns):
        print(f"Пропущены нужные столбцы в {file}")
        continue

    X_new = df[['Time', 'PD_Level']]
    predictions = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]

    df['Predicted_Defect'] = predictions
    df['Defect_Probability'] = proba

    # === СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ ===
    file_name = os.path.basename(file)
    output_name = os.path.join(output_folder, file_name.replace(".csv", "_predicted.csv"))
    df.to_csv(output_name, index=False)
    print(f"Предсказания сохранены: {output_name}")

    # === РИСУЕМ ГРАФИК УРОВНЯ PD И ВЕРОЯТНОСТИ ДЕФЕКТА ===
    plt.figure(figsize=(8, 3))
    plt.plot(df['Time'], df['Defect_Probability'], label='Вероятность дефекта', color='red')
    plt.fill_between(df['Time'], 0, 1,
                     where=df['Predicted_Defect'] == 1,
                     color='orange', alpha=0.2, label='Зона дефекта')
    plt.title(f'Прогнозирование дефектов: {file_name}')
    plt.xlabel('Время')
    plt.ylabel('Уровень / Вероятность')
    plt.legend(loc='upper right')
    plt.tight_layout()
    graph_path = os.path.join(output_folder, f'plot_{idx + 1}.png')
    # plt.show()
    plt.savefig(graph_path)
    plt.close()

    # === СТРОИМ ROC-КРИВУЮ ДЛЯ ОТДЕЛЬНОГО ФАЙЛА ===
    if 'Defect' in df.columns:
        fpr, tpr, _ = roc_curve(df['Defect'], df['Defect_Probability'])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайный классификатор')
        plt.xlabel('Доля ложных срабатываний (FPR)')
        plt.ylabel('Доля истинных срабатываний (TPR)')
        plt.title(f'ROC-кривая: {file_name}')
        plt.legend()
        plt.grid()
        roc_path = os.path.join(output_folder, f'roc_{idx + 1}.png')
        plt.savefig(roc_path)
        plt.close()

        # добавляем в общий список
        all_y_true.extend(df['Defect'])
        all_y_proba.extend(df['Defect_Probability'])

        # === ОЦЕНКА ТОЧНОСТИ ВРЕМЕНИ НАСТУПЛЕНИЯ ДЕФЕКТА ===
        real_defect_idx = df.index[df['Defect'] == 1]
        if len(real_defect_idx) == 0:
            print(f"В файле {file} не найдено реальное наступление дефекта.")
        else:
            real_defect_time = df.loc[real_defect_idx[0], 'Time']

            pred = df['Predicted_Defect'].values
            pred_defect_time = None
            count = 0
            for i in range(len(pred)):
                if pred[i] == 1:
                    count += 1
                    if count == min_consecutive:
                        pred_defect_time = df.loc[i - min_consecutive + 1, 'Time']
                        break
                else:
                    count = 0

            if pred_defect_time is not None:
                time_error = pred_defect_time - real_defect_time
                time_errors.append(time_error)
                print(f"\n=== ОЦЕНКА ВРЕМЕНИ ДЕФЕКТА: {file_name} ===")
                print(f"Реальное время наступления дефекта:      {real_defect_time:.2f}")
                print(f"Предсказанное время (по уставке {min_consecutive}): {pred_defect_time:.2f}")
                print(f"Ошибка определения времени дефекта:      {time_error:.2f} (секунды)")
            else:
                print(f"\n=== ОЦЕНКА ВРЕМЕНИ ДЕФЕКТА: {file_name} ===")
                print(f"Дефект не был обнаружен моделью (нет {min_consecutive} подряд Predicted_Defect=1)")

            # === ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ МЕТОК ДЕФЕКТА ===
            from sklearn.metrics import accuracy_score

            plt.figure(figsize=(14, 4))
            plt.plot(df['Time'], df['Defect'], label='Истинный дефект', drawstyle='steps-post', color='green',
                     linewidth=2)
            plt.plot(df['Time'], df['Predicted_Defect'], label='Предсказанный дефект', drawstyle='steps-post',
                     color='red', alpha=0.7, linewidth=2)

            # Вертикальная линия и подпись: реальное время дефекта
            plt.axvline(real_defect_time, color='green', linestyle='--', linewidth=2, label='Реальный дефект')
            plt.annotate(f'Реальный дефект\n{real_defect_time:.2f} c',
                         xy=(real_defect_time, 1), xycoords='data',
                         xytext=(-60, 25), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", color='green'),
                         fontsize=11, color='green', ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.7))

            # Вертикальная линия и подпись: предсказанное время дефекта
            if pred_defect_time is not None:
                plt.axvline(pred_defect_time, color='red', linestyle='--', linewidth=2, label='Предсказанный дефект')
                plt.annotate(f'Предсказанный дефект\n{pred_defect_time:.2f} c',
                             xy=(pred_defect_time, 1), xycoords='data',
                             xytext=(20, 25), textcoords='offset points',
                             arrowprops=dict(arrowstyle="->", color='red'),
                             fontsize=11, color='red', ha='left', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.7))
                # Подпись с ошибкой
                plt.text(pred_defect_time, 1.12, f'Ошибка: {time_error:.2f} c', color='red', fontsize=12, va='bottom',
                         ha='center')
            else:
                plt.text(df['Time'].iloc[-1], 1.12, 'Дефект не обнаружен моделью', color='red', fontsize=12,
                         va='bottom', ha='right')

            # Для явного задания диапазона оси X:
            plt.xlim(df['Time'].min(), df['Time'].max())

            # Основные метрики
            acc = accuracy_score(df['Defect'], df['Predicted_Defect'])
            plt.title(f"{file_name}\nAUC: {roc_auc:.3f}, Accuracy: {acc:.3f}", fontsize=14)
            plt.xlabel('Время')
            plt.ylabel('Метка дефекта')
            plt.ylim(-0.1, 1.25)
            plt.legend(loc='upper right')
            plt.tight_layout()

            # Формируем имя файла для графика на основе исходного CSV
            base_name = os.path.splitext(file_name)[0]
            if base_name.endswith('_with_defect'):
                base_name = base_name.replace('_with_defect', '')
            compare_path = os.path.join(output_folder, f'compare_defect_{base_name}.png')
            plt.savefig(compare_path)
            # plt.show()
            plt.close()
            print(f"График сравнения меток дефекта сохранён: {compare_path}")

# === СТРОИМ ОБЩУЮ ROC-КРИВУЮ ===
if all_y_true:
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Общая ROC-кривая (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайный классификатор')
    plt.xlabel('Доля ложных срабатываний (FPR)')
    plt.ylabel('Доля истинных срабатываний (TPR)')
    plt.title('Общая ROC-кривая на тестовых данных')
    plt.legend()
    plt.grid()
    global_roc_path = os.path.join(output_folder, 'roc_overall.png')
    plt.savefig(global_roc_path)
    plt.close()

    print(f"Глобальная ROC-кривая сохранена: {global_roc_path}")

# === СТАТИСТИКА ПО ОШИБКЕ ВРЕМЕНИ ДЕФЕКТА ===
if time_errors:
    import numpy as np
    time_errors = np.array(time_errors)
    print("\n=== СТАТИСТИКА ПО ОШИБКЕ ОПРЕДЕЛЕНИЯ ВРЕМЕНИ ДЕФЕКТА ===")
    print(f"Средняя ошибка (сек): {np.mean(time_errors):.2f}")
    print(f"Медианная ошибка (сек): {np.median(time_errors):.2f}")
    print(f"Стандартное отклонение (сек): {np.std(time_errors):.2f}")
    print(f"Минимальная ошибка (сек): {np.min(time_errors):.2f}")
    print(f"Максимальная ошибка (сек): {np.max(time_errors):.2f}")
else:
    print("\nНет данных для статистики по ошибке времени дефекта (дефекты не обнаружены ни в одном файле).")
