# predict_arima_catboost.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# Конфигурация
INPUT_FOLDER = "test_with_mean_amplitude_and_defect"
OUTPUT_FOLDER = "arima_catboost_predictions"  # Новая папка для результатов
MODEL_PATH = "catboost_defect_predictor_mean_amplitude.cbm"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Параметры обработки
TRAIN_SIZE = 150000  # Фиксированное количество строк для обучения

# Загрузка CatBoost модели
catboost_model = joblib.load(MODEL_PATH)

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_FOLDER, file)
    df = pd.read_csv(file_path)
    print(f"\nОбработка файла: {file}")
    print(f"Всего строк: {len(df)}")

    if len(df) <= TRAIN_SIZE:
        print(f"Недостаточно данных для обучения (требуется > {TRAIN_SIZE})")
        continue

    train_df = df.iloc[:TRAIN_SIZE].copy()
    test_df = df.iloc[TRAIN_SIZE:].copy()
    print(f"Обучающих точек: {len(train_df)}")
    print(f"Прогнозируемых точек: {len(test_df)}")

    # Подготовка и очистка данных для ARIMA
    arima_train_data = train_df['Mean_Amplitude'].replace([np.inf, -np.inf], np.nan).dropna()

    # Обучаем модель ARIMA с автоматическим подбором параметров
    try:
        model_arima = auto_arima(
            arima_train_data,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=False,  # Отключаем сезонность
            d=1,             # Порядок интегрирования, обычно 1 для трендовых данных
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True      # Ускоряет подбор параметров
        )
        print("Параметры ARIMA:", model_arima.summary())

        # Прогнозируем на всю длину тестовой выборки
        y_pred_ma = model_arima.predict(n_periods=len(test_df))

    except Exception as e:
        print(f"Ошибка при обучении ARIMA: {e}")
        # В случае ошибки, экстраполируем последнее значение
        y_pred_ma = np.full(len(test_df), train_df['Mean_Amplitude'].iloc[-1])

    # Формирование полного временного ряда
    full_time = np.concatenate([train_df['Time'].values, test_df['Time'].values])
    full_ma = np.concatenate([train_df['Mean_Amplitude'].values, y_pred_ma])

    # Для CatBoost нужны все признаки
    full_pd_level = np.concatenate([
        train_df['PD_Level'].values,
        np.full(len(test_df), train_df['PD_Level'].iloc[-1])
    ])

    # DataFrame для CatBoost
    catboost_data = pd.DataFrame({
        'Time': full_time,
        'PD_Level': full_pd_level,
        'Mean_Amplitude': full_ma
    })

    # Прогнозирование CatBoost
    predictions = catboost_model.predict(catboost_data)
    proba = catboost_model.predict_proba(catboost_data)[:, 1]
    defect_percentage = 100 * np.mean(predictions)
    print(f"Процент времени с дефектом: {defect_percentage:.2f}%")

    # Сохранение результатов
    result_df = pd.DataFrame({
        'Time': full_time,
        'Mean_Amplitude': full_ma,
        'Defect_Probability': proba,
        'Predicted_Defect': predictions,
        'Is_Predicted': [0] * len(train_df) + [1] * len(test_df)
    })
    output_path = os.path.join(OUTPUT_FOLDER, f"predicted_{file}")
    result_df.to_csv(output_path, index=False)
    print(f"Результаты сохранены в {output_path}")

    # Визуализация 1: Прогноз ARIMA
    plt.figure(figsize=(16, 8))
    plt.plot(train_df['Time'], train_df['Mean_Amplitude'], 'b-', label='Исторические данные', alpha=0.7)
    plt.plot(test_df['Time'], test_df['Mean_Amplitude'], 'g.', label='Реальные значения (будущее)', markersize=4, alpha=0.6)
    plt.plot(test_df['Time'], y_pred_ma, 'r--', label='Прогноз ARIMA', linewidth=2)
    plt.axvline(x=test_df['Time'].iloc[0], color='k', linestyle='--', label='Начало прогноза')
    plt.title(f'Прогноз ARIMA для {file}')
    plt.xlabel('Время')
    plt.ylabel('Средняя амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path1 = os.path.join(OUTPUT_FOLDER, f"arima_{file.replace('.csv', '.png')}")
    plt.savefig(plot_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"График ARIMA сохранен: {plot_path1}")

    # Визуализация 2: Прогноз дефекта
    plt.figure(figsize=(16, 8))
    plt.plot(full_time, full_ma, 'b-', label='Средняя амплитуда', linewidth=2)
    plt.axvline(x=test_df['Time'].iloc[0], color='k', linestyle='--', alpha=0.7, label='Начало прогноза')
    plt.xlabel('Время', fontsize=12)
    plt.ylabel('Средняя амплитуда', fontsize=12)
    plt.grid(True, alpha=0.3)

    ax2 = plt.twinx()
    ax2.plot(full_time, proba, 'r-', label='Вероятность дефекта', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Вероятность дефекта', fontsize=12)
    ax2.set_ylim(0, 1)

    defect_start = None
    for i, pred in enumerate(predictions):
        if pred == 1 and defect_start is None:
            defect_start = full_time[i]
        elif pred == 0 and defect_start is not None:
            plt.axvspan(defect_start, full_time[i - 1], alpha=0.2, color='red')
            defect_start = None
    if defect_start is not None:
        plt.axvspan(defect_start, full_time[-1], alpha=0.2, color='red')

    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title(f'Прогноз дефекта для {file}', fontsize=14)
    plot_path2 = os.path.join(OUTPUT_FOLDER, f"defect_{file.replace('.csv', '.png')}")
    plt.savefig(plot_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"График дефекта сохранен: {plot_path2}")

print("\nОбработка всех файлов завершена. Результаты в", OUTPUT_FOLDER)
