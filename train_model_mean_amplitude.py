import pandas as pd
import glob
import joblib
from catboost import CatBoostClassifier

# Конфигурация
INPUT_FOLDER = "train_with_mean_amplitude_and_defect"
MODEL_PATH = "catboost_mean_amplitude_model.cbm"

# Загрузка и объединение данных
files = glob.glob(f"{INPUT_FOLDER}/*.csv")
df_list = [pd.read_csv(file) for file in files]
train_df = pd.concat(df_list, ignore_index=True)

# Подготовка данных
X = train_df[['Time', 'PD_Level', 'Mean_Amplitude']]
y = train_df['Defect']

# Обучение модели
model = CatBoostClassifier(
    verbose=0
)
model.fit(X, y)

# Сохранение модели
joblib.dump(model, MODEL_PATH)
print(f"Модель CatBoost обучена и сохранена в {MODEL_PATH}")
