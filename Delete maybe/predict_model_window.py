import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
import numpy as np

# НЕ РАБОТАЕТ


# === 1. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
model = joblib.load('catboost_defect_predictor_with_window.cbm')

# === 2. ПОИСК CSV-ФАЙЛОВ ДЛЯ ПРЕДСКАЗАНИЯ ===
predict_files = glob.glob("test_with_defect/*.csv")
output_folder = "predicted"
os.makedirs(output_folder, exist_ok=True)

# === ДЛЯ ОБЩЕЙ ROC-КРИВОЙ ===
all_y_true = []
all_y_proba = []

# === 3. ОБРАБОТКА КАЖДОГО ФАЙЛА ===
window_size = 10  # размер окна, например 10 временных точек

for idx, file in enumerate(predict_files):
    df = pd.read_csv(file)

    if not {'Time', 'PD_Level'}.issubset(df.columns):
        print(f"Пропущены нужные столбцы в {file}")
        continue

    # Преобразуем данные в окна
    X_new = []
    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size:i]  # скользящее окно
        features = window[['Time', 'PD_Level']].values.flatten()  # превращаем окно в плоский массив
        X_new.append(features)

    X_new = np.array(X_new)

    # Предсказания и вероятности
    predictions = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]

    df['Predicted_Defect'] = [None] * window_size + list(predictions)
    df['Defect_Probability'] = [None] * window_size + list(proba)

    # === СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ ===
    file_name = os.path.basename(file)
    output_name = os.path.join(output_folder, file_name.replace(".csv", "_predicted.csv"))
    df.to_csv(output_name, index=False)
    print(f"Предсказания сохранены: {output_name}")

    # === РИСУЕМ ГРАФИК PD И ПРОБА ===
    plt.figure(figsize=(8, 3))
    plt.plot(df['Time'], df['PD_Level'], label='PD Level')
    plt.plot(df['Time'], df['Defect_Probability'], label='Defect Probability', color='red')
    plt.fill_between(df['Time'], 0, 1,
                     where=df['Predicted_Defect'] == 1,
                     color='orange', alpha=0.2, label='Defect Zone')
    plt.title(f'Defect Prediction: {file_name}')
    plt.xlabel('Time')
    plt.ylabel('Level / Probability')
    plt.legend(loc='upper right')
    plt.tight_layout()
    graph_path = os.path.join(output_folder, f'plot_{idx + 1}.png')
    plt.savefig(graph_path)
    plt.close()

    # === СТРОИМ ROC ДЛЯ ОТДЕЛЬНОГО ФАЙЛА ===
    if 'Defect' in df.columns:
        fpr, tpr, _ = roc_curve(df['Defect'], df['Defect_Probability'])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {file_name}')
        plt.legend()
        plt.grid()
        roc_path = os.path.join(output_folder, f'roc_{idx + 1}.png')
        plt.savefig(roc_path)
        plt.close()

        # добавляем в общий список
        all_y_true.extend(df['Defect'])
        all_y_proba.extend(df['Defect_Probability'])

# === 4. СТРОИМ ОБЩУЮ ROC-КРИВУЮ ===
if all_y_true:
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Global ROC (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve on test_with_defect')
    plt.legend()
    plt.grid()
    global_roc_path = os.path.join(output_folder, 'roc_overall.png')
    plt.savefig(global_roc_path)
    plt.close()

    print(f"Глобальная ROC-кривая сохранена: {global_roc_path}")
