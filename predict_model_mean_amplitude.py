# predict_model_mean_amplitude.py

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
import numpy as np

# === 1. ЗАГРУЗКА НОВОЙ МОДЕЛИ ===
MODEL_PATH = 'catboost_defect_predictor_mean_amplitude.cbm'
model = joblib.load(MODEL_PATH)

# === 2. НАСТРОЙКА ПУТЕЙ ===
INPUT_TEST_FOLDER = "test_with_mean_amplitude_and_defect"  # Новая папка с тестовыми данными
OUTPUT_FOLDER = "predicted_mean_amplitude"  # Новая папка для результатов
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === 3. ПАРАМЕТРЫ ОБРАБОТКИ ===
MIN_CONSECUTIVE = 200  # Уставка для последовательных Predicted_Defect=1
WINDOW_SIZE = 15000  # Должно соответствовать значению при подготовке данных

# === 4. ОБРАБОТКА ФАЙЛОВ ===
all_y_true = []
all_y_proba = []
time_errors = []

for file in glob.glob(f"{INPUT_TEST_FOLDER}/*.csv"):
    df = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]

    # Проверка необходимых столбцов
    required_columns = {'Time', 'PD_Level', 'Mean_Amplitude', 'Defect'}
    if not required_columns.issubset(df.columns):
        print(f"Пропущены нужные столбцы в {file}")
        continue

    # Формирование признаков
    X_new = df[['Time', 'PD_Level', 'Mean_Amplitude']]

    # Предсказание
    predictions = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]

    # Сохранение результатов
    output_df = df.copy()
    output_df['Predicted_Defect'] = predictions
    output_df['Defect_Probability'] = proba
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_predicted.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Предсказания сохранены: {output_path}")

    # Визуализация вероятности дефекта
    plt.figure(figsize=(10, 4))
    plt.plot(output_df['Time'], output_df['Defect_Probability'], color='darkred', label='Вероятность дефекта')
    plt.fill_between(output_df['Time'], 0, 1,
                     where=output_df['Predicted_Defect'] == 1,
                     color='salmon', alpha=0.3, label='Зона дефекта')
    plt.title(f'Прогноз дефектов: {base_name}')
    plt.xlabel('Время, сек')
    plt.ylabel('Вероятность')
    plt.legend()
    plot_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_probability.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Оценка времени дефекта
    real_defect_time = output_df.loc[output_df['Defect'].idxmax(), 'Time']

    pred_defect_time = None
    counter = 0
    for idx, row in output_df.iterrows():
        if row['Predicted_Defect'] == 1:
            counter += 1
            if counter >= MIN_CONSECUTIVE:
                pred_defect_time = row['Time'] - (WINDOW_SIZE / 2 * 1e-6)  # Коррекция времени для центра окна
                break
        else:
            counter = 0

    # Запись ошибки
    if pred_defect_time is not None:
        time_error = pred_defect_time - real_defect_time
        time_errors.append(time_error)
        print(f"\nОшибка времени для {base_name}: {time_error:.2f} сек")

# === 5. ГЛОБАЛЬНАЯ ОЦЕНКА ===
if time_errors:
    print("\n=== СТАТИСТИКА ОШИБОК ВРЕМЕНИ ===")
    print(f"Средняя ошибка: {np.mean(time_errors):.2f} сек")
    print(f"Медианная ошибка: {np.median(time_errors):.2f} сек")
    print(f"Стандартное отклонение: {np.std(time_errors):.2f} сек")

# Генерация ROC-кривой
if all_y_true:
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Общая ROC-кривая')
    plt.legend()
    roc_path = os.path.join(OUTPUT_FOLDER, 'roc_curve.png')
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"\nROC-кривая сохранена: {roc_path}")
