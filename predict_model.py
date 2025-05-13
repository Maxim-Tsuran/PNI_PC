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
output_folder = "predicted"
os.makedirs(output_folder, exist_ok=True)

# === ДЛЯ ОБЩЕЙ ROC-КРИВОЙ ===
all_y_true = []
all_y_proba = []

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