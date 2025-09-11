# train_model.py
import pandas as pd
import glob
from catboost import CatBoostClassifier
import joblib

# === 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ CSV-ФАЙЛОВ ===
train_files = glob.glob("train_with_defect/*.csv")
df_list = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(df_list, ignore_index=True)

# === 2. ПОДГОТОВКА ДАННЫХ ===
X = train_df[['Time', 'PD_Level']]
y = train_df['Defect']

# === 3. ОБУЧЕНИЕ МОДЕЛИ ===
model = CatBoostClassifier(verbose=0)
model.fit(X, y)

# === 4. СОХРАНЕНИЕ МОДЕЛИ ===
joblib.dump(model, 'catboost_defect_predictor.cbm')
print("Модель обучена и сохранена в файл: catboost_defect_predictor.cbm")
