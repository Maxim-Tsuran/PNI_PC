# train_model_mean_amplitude.py

import pandas as pd
import glob
from catboost import CatBoostClassifier
import joblib
import os

# === 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ CSV-ФАЙЛОВ ===
train_files = glob.glob("train_with_mean_amplitude_and_defect/*.csv")
if not train_files:
    raise FileNotFoundError("Нет файлов для обучения в папке train_with_mean_amplitude_and_defect")

df_list = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(df_list, ignore_index=True)

# === 2. ПОДГОТОВКА ДАННЫХ ===
# Используем все три признака
X = train_df[['Time', 'PD_Level', 'Mean_Amplitude']]
y = train_df['Defect']

# === 3. ОБУЧЕНИЕ МОДЕЛИ ===
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X, y)

# === 4. СОХРАНЕНИЕ МОДЕЛИ ===
model_path = 'catboost_defect_predictor_mean_amplitude.cbm'
joblib.dump(model, model_path)
print(f"Модель обучена и сохранена в файл: {model_path}")
