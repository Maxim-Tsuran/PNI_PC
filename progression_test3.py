import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pmdarima as pm
from pmdarima import model_selection

# 1. Загрузка и предобработка данных (как и раньше)
def load_and_preprocess(file_path, sample_rate=400, train_size=150000):
    """Загружает данные, выбирает столбцы, прореживает и разделяет на обучающую и тестовую выборки."""
    df = pd.read_csv(file_path)
    df = df[["Time", "PD_Level"]]
    df = df.iloc[::sample_rate, :]  # Выбираем каждую sample_rate-ую строку
    df = df.reset_index(drop=True) # Переиндексация после прореживания
    train_data = df[:train_size // sample_rate]
    test_data = df[train_size // sample_rate:]
    return train_data, test_data

# 2. Обучение модели ARIMA (с auto_arima)
def train_arima(train_data):
    """Обучает модель ARIMA с автоматическим подбором параметров."""
    model = pm.auto_arima(train_data["PD_Level"],
                          start_p=0, start_q=0,
                          max_p=5, max_q=5,
                          m=1,  # Сезонности нет, поэтому m=1
                          d=None,   # Позволяем auto_arima определить d
                          seasonal=False,  # Сезонность отключена
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    return model

# 3. Прогнозирование (итеративное)
def iterative_forecast_arima(model, test_data):
    """Прогнозирует значения итеративно, используя предыдущий прогноз для следующего шага."""
    predictions = []
    history = list(train_data["PD_Level"])
    for _ in range(len(test_data)):
        model_fit = model.fit(history)  # Переобучаем модель на истории
        output = model.predict(n_periods=1)[0]
        predictions.append(output)
        history.append(output)
    return predictions


# 4. Оценка результатов (как и раньше)
def evaluate_forecast(test_data, predictions):
    """Вычисляет метрики качества прогнозирования."""
    mse = mean_squared_error(test_data["PD_Level"], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data["PD_Level"], predictions)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return mse, rmse, mae

# 5. Визуализация (как и раньше)
def plot_results(train_data, test_data, predictions):
    """Строит графики."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_data["PD_Level"], label="Обучающие данные")
    plt.plot(test_data.index, test_data["PD_Level"], label="Реальные значения (тест)")
    plt.plot(test_data.index, predictions, label="Прогнозы")
    plt.xlabel("Индекс (после прореживания)")
    plt.ylabel("PD_Level")
    plt.title("Прогнозирование PD_Level с помощью ARIMA")
    plt.legend()
    plt.show()


# ---  Основной код  ---
file_path = r'C:\Users\tsura\PycharmProjects\PNI_PC\test_with_defect\U_PD5_with_defect.csv' # Замените на имя вашего файла

# Загрузка, предобработка и разделение данных
train_data, test_data = load_and_preprocess(file_path, sample_rate=400)

# Обучение модели ARIMA
model = train_arima(train_data)

# Прогнозирование
predictions = iterative_forecast_arima(model, test_data)

# Оценка результатов
mse, rmse, mae = evaluate_forecast(test_data, predictions)

# Визуализация
plot_results(train_data, test_data, predictions)