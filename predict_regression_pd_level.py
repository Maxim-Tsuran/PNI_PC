import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import os

# === НАСТРОЙКИ ===
INPUT_FILE = r'C:\Users\tsura\PycharmProjects\PNI_PC\test_with_defect\U_PD5_with_defect.csv'
HISTORY_SIZE = 1000                   # количество предыдущих значений для предсказания
TRAIN_SIZE = 150000                 # сколько строк использовать для обучения
MODEL_PATH = 'catboost_pd_regression_predictor.cbm'
SAMPLE_STEP = 400                    # берем каждые 400-е точки

# === 1. ЗАГРУЗКА ДАННЫХ И ПРОРЕЖИВАНИЕ ===
df = pd.read_csv(INPUT_FILE)
assert 'PD_Level' in df.columns and 'Time' in df.columns, "Отсутствуют нужные столбцы"

# Прореживаем данные - берем каждые SAMPLE_STEP точек
df_sampled = df.iloc[::SAMPLE_STEP].reset_index(drop=True)
print(f"Всего точек после прореживания: {len(df_sampled)}")

# === 2. СОЗДАНИЕ ПРИЗНАКОВ НА ОСНОВЕ ОКНА ИСТОРИИ ===
def create_features(data, history):
    X, y = [], []
    for i in range(history, len(data)):
        X.append(data[i-history:i])  # предыдущие значения PD_Level
        y.append(data[i])            # текущее значение
    return np.array(X), np.array(y)

pd_series = df_sampled['PD_Level'].values
X_all, y_all = create_features(pd_series, HISTORY_SIZE)

# === 3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКУ ===
train_size = min(TRAIN_SIZE // SAMPLE_STEP, len(X_all))  # адаптируем размер под прореженные данные
X_train = X_all[:train_size]
y_train = y_all[:train_size]

X_test = X_all[train_size:]
y_test = y_all[train_size:]

print(f"Обучающих примеров: {len(X_train)}")
print(f"Тестовых примеров: {len(X_test)}")

# === 4. ОБУЧЕНИЕ МОДЕЛИ ===
model = CatBoostRegressor(verbose=0)
model.fit(X_train, y_train)

# === 5. СОХРАНЕНИЕ МОДЕЛИ ===
model.save_model(MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")

# === 6. ПРЕДСКАЗАНИЕ ===
y_pred = model.predict(X_test)

# === 7. СОХРАНЕНИЕ И СРАВНЕНИЕ ===

# Correctly slice the Time series to match the length of y_test and y_pred
comparison_df = pd.DataFrame({
    'Time': df_sampled['Time'].values[train_size + HISTORY_SIZE:],  # Adjusted slicing
    'PD_Level_True': y_test,
    'PD_Level_Predicted': y_pred
})
comparison_df.to_csv('pd_forecast_comparison.csv', index=False)

# === 8. ПОСТРОЕНИЕ ГРАФИКА ===
plt.figure(figsize=(12, 4))
plt.plot(comparison_df['Time'], comparison_df['PD_Level_True'],
         label='Истинный PD_Level', alpha=0.7, linewidth=1)
plt.plot(comparison_df['Time'], comparison_df['PD_Level_Predicted'],
         label='Предсказанный PD_Level', alpha=0.7, linewidth=1)
plt.title(f"Сравнение PD_Level (каждые {SAMPLE_STEP}-е точки)")
plt.xlabel("Time")
plt.ylabel("PD_Level")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('pd_forecast_plot.png')
plt.close()

print("График сохранён в pd_forecast_plot.png")