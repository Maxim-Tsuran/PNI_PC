import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error

# === Параметры ===
filename = r'C:\Users\tsura\PycharmProjects\PNI_PC\test_with_defect\U_PD5_with_defect.csv'
skip_rate = 100  # больше данных после downsampling
train_limit = 150000  # чтобы хватало строк даже после skip
history = 200  # вместо 1000
n_freqs = 10

# === Загрузка и уменьшение объема ===
df = pd.read_csv(filename)
df_down = df.iloc[::skip_rate, :].reset_index(drop=True)

# Делим по train_limit
train_df = df_down[df_down.index < train_limit // skip_rate]
test_df = df_down[df_down.index >= train_limit // skip_rate]

X_train_time = train_df['Time'].values
y_train_full = train_df['PD_Level'].values
X_test_time = test_df['Time'].values
y_test_full = test_df['PD_Level'].values

# === Получаем частоты из Фурье ===
def select_top_freqs(time_array, signal_array, n_freqs=12):
    N = len(signal_array)
    T = np.mean(np.diff(time_array))
    yf = fft(signal_array)
    xf = fftfreq(N, T)
    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])
    top_indices = np.argsort(yf)[-n_freqs:]
    top_freqs = xf[top_indices]
    return sorted(top_freqs)

top_freqs = select_top_freqs(X_train_time, y_train_full, n_freqs=n_freqs)
print("Выбранные частоты:", top_freqs)

# === Построение признаков ===
def create_features_with_time_and_sin(data, time, history, freqs):
    X, y = [], []
    for i in range(history, len(data)):
        pd_window = data[i-history:i]
        features = list(pd_window)
        features.append(time[i])
        for f in freqs:
            features.append(np.sin(2 * np.pi * f * time[i]) * time[i])
            features.append(np.cos(2 * np.pi * f * time[i]) * time[i])
        X.append(features)
        y.append(data[i])
    return np.array(X), np.array(y)

# Обучение
X_train, y_train = create_features_with_time_and_sin(
    y_train_full, X_train_time, history, top_freqs)

# Тест
X_test, y_test = create_features_with_time_and_sin(
    np.concatenate([y_train_full[-history:], y_test_full]),  # добавим хвост
    np.concatenate([X_train_time[-history:], X_test_time]),
    history, top_freqs)

print(f"Размер обучающей выборки: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Размер тестовой выборки: X_test = {X_test.shape}, y_test = {y_test.shape}")
# === Обучение модели ===
model = CatBoostRegressor(verbose=0, iterations=500, learning_rate=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Оценка ===
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2e}")

# === График ===
plt.figure(figsize=(16, 8))
plt.plot(X_train_time[history:], y_train, label='Обучающая выборка', color='blue', alpha=0.5)
plt.plot(X_test_time, y_test, label='Тестовая выборка', color='green', alpha=0.8)
plt.plot(X_test_time, y_pred, label='Прогноз', color='red', linestyle='--', linewidth=2)

plt.axvline(x=X_test_time[0], color='black', linestyle=':', linewidth=2)
plt.text(X_test_time[0], max(y_test)*0.95, ' Начало прогноза',
         ha='left', va='top', backgroundcolor='white', fontsize=10)

plt.xlabel("Время")
plt.ylabel("PD_Level")
plt.title(f"CatBoost + Синусоиды по времени (MSE: {mse:.2e})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('pd_regression_plot.png')
plt.close()
