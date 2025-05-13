import pandas as pd
import glob
from catboost import CatBoostClassifier
import joblib
import numpy as np

# НЕ РАБОТАЕТ



# === 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ CSV-ФАЙЛОВ ===
train_files = glob.glob("train_with_defect/*.csv")
df_list = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(df_list, ignore_index=True)

# === 2. ПОДГОТОВКА ДАННЫХ С ИСПОЛЬЗОВАНИЕМ СКОЛЬЗЯЩИХ ОКОН ===
window_size = 1000  # размер окна, например 10 временных точек

X = []
y = []

# Проходим по всем строкам данных, начиная с позиции после окна
for i in range(window_size, len(train_df)):
    window = train_df.iloc[i - window_size:i]  # скользящее окно данных
    # Используем все признаки из окна
    features = window[['Time', 'PD_Level']].values.flatten()  # превращаем окно в плоский массив
    target = train_df.iloc[i]['Defect']  # метка дефекта для следующей точки (после окна)
    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

# === 3. ОБУЧЕНИЕ МОДЕЛИ ===
model = CatBoostClassifier(verbose=0)
model.fit(X, y)

# === 4. СОХРАНЕНИЕ МОДЕЛИ ===
joblib.dump(model, 'catboost_defect_predictor_with_window.cbm')
print("Модель обучена и сохранена в файл: catboost_defect_predictor_with_window.cbm")
