import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, fftfreq

# ВРОДЕ РАБОТАЕТ. ДАЛЕЕ НАДО ПРОВЕРИТЬ РАБОТУ КЛАССИФИКАТОРА НА ПРОГНОЗЕ, КОТОРЫЙ ПОЛУЧИЛСЯ ЗДЕСЬ
# === Параметры ===
filename = r'C:\Users\tsura\PycharmProjects\PNI_PC\test_with_defect\U_PD5_with_defect.csv'
skip_rate = 2000
train_limit = 100000

# === Загрузка и уменьшение объема данных ===
df = pd.read_csv(filename)
df_down = df.iloc[::skip_rate, :].reset_index(drop=True)

# Разделение
train_df = df_down[df_down.index < train_limit // skip_rate]
test_df = df_down[df_down.index >= train_limit // skip_rate]

X_train_time = train_df['Time'].values
y_train = train_df['PD_Level'].values
X_test_time = test_df['Time'].values
y_test = test_df['PD_Level'].values


def select_top_freqs(time_array, signal_array, n_freqs=12):
    N = len(signal_array)
    T = np.mean(np.diff(time_array))  # средний шаг времени
    yf = fft(signal_array)
    xf = fftfreq(N, T)

    # Оставляем только положительные частоты
    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])

    # Сортировка по амплитуде, выбираем top-N частот
    top_indices = np.argsort(yf)[-n_freqs:]
    top_freqs = xf[top_indices]
    return sorted(top_freqs)

# === Генерация синусоидальных признаков ===
def build_features(time_array, freqs):
    features = [time_array]
    for f in freqs:
        features.append(np.sin(2 * np.pi * f * time_array) * time_array)  # растущая амплитуда
        features.append(np.cos(2 * np.pi * f * time_array) * time_array)
    return np.stack(features, axis=1)

# Частоты подбираем вручную или через спектр — пока зафиксируем:
freqs = select_top_freqs(X_train_time, y_train, n_freqs=12)
print("Автоматически выбранные частоты:", freqs)

#freqs = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000] # Вручную задать можно
X_train = build_features(X_train_time, freqs)
X_test = build_features(X_test_time, freqs)

# === Обучение модели ===
model = Ridge(alpha=1e-6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Оценка ===
mse = mean_squared_error(y_test, y_pred)
# Абсолютные ошибки
abs_errors = np.abs(y_test - y_pred)
mae = np.mean(abs_errors)
max_err = np.max(abs_errors)
mse = mean_squared_error(y_test, y_pred)  # уже было
rmse = np.sqrt(mse)

# В процентах — относительно среднего значения тестовых данных
mean_y = np.mean(np.abs(y_test))
mae_pct = mae / mean_y * 100
max_err_pct = max_err / mean_y * 100
rmse_pct = rmse / mean_y * 100

# Вывод ошибок
print(f"\n=== ОЦЕНКА КАЧЕСТВА ПРОГНОЗА ===")
print(f"Среднеквадратичная ошибка (MSE):     {mse:.2e} (ед. изм.)")
print(f"Среднекорневая ошибка (RMSE):        {rmse:.4f} ({rmse_pct:.2f}%)")
print(f"Средняя абсолютная ошибка (MAE):     {mae:.4f} ({mae_pct:.2f}%)")
print(f"Максимальная ошибка:                 {max_err:.4f} ({max_err_pct:.2f}%)")

# === График ===
plt.figure(figsize=(16, 8))

# Обучающая часть
plt.plot(X_train_time, y_train,
         label="Обучающие данные", color='blue', alpha=0.5)

# Тестовая часть и прогноз
plt.plot(X_test_time, y_test,
         label="Тестовые данные", color='green', alpha=0.8, linewidth=1.5)
plt.plot(X_test_time, y_pred,
         label="Прогноз", color='red', linestyle='--', linewidth=2)

# Настройки графика (увеличенные шрифты)
plt.xlabel("Время", fontsize=24)  # Было 12, стало 24
plt.ylabel("Уровень ЧР", fontsize=24)  # Было 12, стало 24
plt.title(f"Прогноз уровня ЧР (шаг прореживания: {skip_rate})", fontsize=28)  # Было 14, стало 28

# Вертикальная линия разделения
split_time = X_test_time[0]  # Первый момент тестовой выборки
plt.axvline(x=split_time, color='black', linestyle=':', linewidth=2, alpha=0.7)
plt.text(split_time, max(y_test)*0.95, ' Начало прогноза',
         ha='left', va='top', backgroundcolor='white', fontsize=20)  # Было 10, стало 20

# Легенда (крупный шрифт)
plt.legend(loc='upper right', fontsize=20)  # Было 10, стало 20
plt.grid(True, linestyle='--', alpha=0.6)

# Увеличение размера цифр на осях
plt.tick_params(axis='both', which='major', labelsize=18)  # Размер цифр на делениях осей
# Увеличение размера цифр на осях X и Y
plt.xticks(fontsize=18)  # Размер цифр на оси X
plt.yticks(fontsize=18)  # Размер цифр на оси Y
# Масштабирование осей
y_padding = 0.05 * (max(y_test) - min(y_test))
plt.ylim(min(y_test)-y_padding, max(y_test)+y_padding)

# Деления на оси X
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

# Заливка для областей
plt.axvspan(min(X_train_time), split_time, color='blue', alpha=0.05, label='Обучающая выборка')
plt.axvspan(split_time, max(X_test_time), color='red', alpha=0.05, label='Тестовая выборка')

# Обновление легенды с крупным шрифтом
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right', fontsize=20)  # Было 10, стало 20

plt.tight_layout()
plt.savefig('Ridge_Regression_sincos_regress_improved.png', dpi=300, bbox_inches='tight')
plt.show()





# === Новый блок: расчет средних амплитуд по окну ===
# Для построения графика средних значений амплитуд, только выше оси X, по графику можно понять
window_size = 25  # Ширина окна (можно подобрать). Чем больше, тем меньше погрешность и больше ретроспективы требуется
def moving_average_abs(y, window):
    return np.convolve(np.abs(y), np.ones(window)/window, mode='valid')

ma_test = moving_average_abs(y_test, window_size)
ma_pred = moving_average_abs(y_pred, window_size)
ma_time = X_test_time[window_size//2:-(window_size//2)] if window_size % 2 == 1 else X_test_time[window_size//2-1:-(window_size//2)]


# === Оценка качества прогноза для средних амплитуд ===
ma_abs_errors = np.abs(ma_test - ma_pred)
ma_mae = np.mean(ma_abs_errors)
ma_max_err = np.max(ma_abs_errors)
ma_mse = np.mean((ma_test - ma_pred) ** 2)
ma_rmse = np.sqrt(ma_mse)

# В процентах — относительно среднего значения скользящей средней теста
ma_mean_y = np.mean(np.abs(ma_test))
ma_mae_pct = ma_mae / ma_mean_y * 100
ma_max_err_pct = ma_max_err / ma_mean_y * 100
ma_rmse_pct = ma_rmse / ma_mean_y * 100
ma_mse_pct = (ma_rmse / ma_mean_y) * 100  # MSE в процентах через RMSE


print(f"\n=== ОЦЕНКА КАЧЕСТВА ПРОГНОЗА (средняя амплитуда) ===")
print(f"Среднеквадратичная ошибка (MSE):     {ma_mse:.2e} (ед. изм.), {ma_mse_pct:.2f}%")
print(f"Среднекорневая ошибка (RMSE):        {ma_rmse:.2e} ({ma_rmse_pct:.2f}%)")
print(f"Средняя абсолютная ошибка (MAE):     {ma_mae:.2e} ({ma_mae_pct:.2f}%)")
print(f"Максимальная ошибка:                 {ma_max_err:.2e} ({ma_max_err_pct:.2f}%)")


# === График средних амплитуд ===
plt.figure(figsize=(16, 8))
plt.plot(ma_time, ma_test, label='Средняя амплитуда (тест)', color='green', linewidth=2)
plt.plot(ma_time, ma_pred, label='Средняя амплитуда (прогноз)', color='red', linestyle='--', linewidth=2)
plt.xlabel("Время", fontsize=24)
plt.ylabel("Средняя амплитуда ЧР", fontsize=24)
plt.title(f"Средняя амплитуда ЧР (скользящее среднее, окно={window_size})", fontsize=28)
plt.legend(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Ridge_Regression_mean_Ampl.png', dpi=300, bbox_inches='tight')
plt.show()
