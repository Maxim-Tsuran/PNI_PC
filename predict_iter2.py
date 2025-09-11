import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# === НАСТРОЙКИ ===
INPUT_FILE = r'C:\Users\tsura\PycharmProjects\PNI_PC\test_with_defect\U_PD5_with_defect.csv'
HISTORY_SIZE = 1000                  # количество предыдущих значений для предсказания
TRAIN_SIZE = 150000                  # сколько строк использовать для обучения
MODEL_PATH = 'catboost_pd_regression_predictor.cbm'

# === 1. ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv(INPUT_FILE)
assert 'PD_Level' in df.columns and 'Time' in df.columns, "Отсутствуют нужные столбцы"

# === 2. СОЗДАНИЕ ПРИЗНАКОВ НА ОСНОВЕ ОКНА ИСТОРИИ ===
def create_features(data, history):
    X, y = [], []
    for i in range(history, len(data)):
        X.append(data[i-history:i])
        y.append(data[i])
    return np.array(X), np.array(y)

pd_series = df['PD_Level'].values
X_all, y_all = create_features(pd_series, HISTORY_SIZE)

# === 3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКУ ===
X_train = X_all[:TRAIN_SIZE - HISTORY_SIZE]
y_train = y_all[:TRAIN_SIZE - HISTORY_SIZE]

# === 4. ОБУЧЕНИЕ МОДЕЛИ ===
model = CatBoostRegressor(verbose=0)
model.fit(X_train, y_train)

# === 5. СОХРАНЕНИЕ МОДЕЛИ ===
model.save_model(MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")

# === 6. МНОГОШАГОВОЕ ПРЕДСКАЗАНИЕ С НАКОПЛЕНИЕМ ===
def multi_step_predict(model, initial_window, steps):
    history = list(initial_window)
    predictions = []

    for _ in range(steps):
        input_data = np.array(history[-HISTORY_SIZE:]).reshape(1, -1)
        next_pred = model.predict(input_data)[0]
        predictions.append(next_pred)
        history.append(next_pred)

    return np.array(predictions)

# Последние значения перед границей TRAIN_SIZE
initial_window = pd_series[TRAIN_SIZE - HISTORY_SIZE:TRAIN_SIZE]
future_steps = len(df) - TRAIN_SIZE

# Прогнозирование
y_pred = multi_step_predict(model, initial_window, future_steps)

# === 7. СОХРАНЕНИЕ И СРАВНЕНИЕ ===
y_true = df['PD_Level'].values[TRAIN_SIZE:]
comparison_df = pd.DataFrame({
    'Time': df['Time'].values[TRAIN_SIZE:],
    'PD_Level_True': y_true,
    'PD_Level_Predicted': y_pred
})
comparison_df.to_csv('pd_forecast_comparison.csv', index=False)
print("Сравнение сохранено в pd_forecast_comparison.csv")

# === 8. ПОСТРОЕНИЕ ГРАФИКА ===
plt.figure(figsize=(12, 4))
plt.plot(comparison_df['Time'], comparison_df['PD_Level_True'], label='Истинный PD_Level', alpha=0.7)
plt.plot(comparison_df['Time'], comparison_df['PD_Level_Predicted'], label='Предсказанный PD_Level', alpha=0.7)
plt.title("Многошаговое предсказание PD_Level (autoregressive)")
plt.xlabel("Time")
plt.ylabel("PD_Level")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('pd_forecast_plot.png')
plt.close()

print("График сохранён в pd_forecast_plot.png")
