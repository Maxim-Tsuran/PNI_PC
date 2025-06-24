# predict_linear_regression_catboost.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Конфигурация
INPUT_FOLDER = "test_with_mean_amplitude_and_defect"
OUTPUT_FOLDER = "linear_regression_catboost_predictions"  # Новая папка для результатов
MODEL_PATH = "catboost_mean_amplitude_model.cbm"
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

    # Подготовка данных для Линейной Регрессии
    # X - это время, y - это средняя амплитуда
    X_train = train_df[['Time']].values
    y_train = train_df['Mean_Amplitude'].values

    # Обучаем простую модель Линейной Регрессии
    try:
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)

        # Прогнозируем на временных метках тестовой выборки
        X_test = test_df[['Time']].values
        y_pred_ma = model_lr.predict(X_test)

        # Убедимся, что прогноз не уходит в отрицательные значения
        y_pred_ma[y_pred_ma < 0] = 0

    except Exception as e:
        print(f"Ошибка при обучении Линейной Регрессии: {e}")
        # В случае ошибки, экстраполируем последнее известное значение
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

    # Визуализация 1: Прогноз Линейной Регрессии
    plt.figure(figsize=(16, 8))
    plt.plot(train_df['Time'], train_df['Mean_Amplitude'], 'b-', label='Исторические данные', alpha=0.7)
    plt.plot(test_df['Time'], test_df['Mean_Amplitude'], 'g.', label='Реальные значения (будущее)', markersize=4, alpha=0.6)
    plt.plot(test_df['Time'], y_pred_ma, 'r--', label='Прогноз (Линейная Регрессия)', linewidth=2)
    plt.axvline(x=test_df['Time'].iloc[0], color='k', linestyle='--', label='Начало прогноза')
    plt.title(f'Прогноз Линейной Регрессии для {file}')
    plt.xlabel('Время')
    plt.ylabel('Средняя амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path1 = os.path.join(OUTPUT_FOLDER, f"linear_regression_{file.replace('.csv', '.png')}")
    plt.savefig(plot_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"График Линейной Регрессии сохранен: {plot_path1}")

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

    # === Новый блок: сравнение времени наступления дефекта ===

    # 1. Поиск времени реального дефекта
    real_defect = np.concatenate([train_df['Defect'].values, test_df['Defect'].values])
    real_defect_idx = np.where(real_defect == 1)[0]
    if len(real_defect_idx) == 0:
        real_defect_time = None
        print("В файле не найдено реальное наступление дефекта.")
    else:
        real_defect_time = full_time[real_defect_idx[0]]

    # 2. Поиск времени предсказанного дефекта (по уставке подряд идущих единиц)
    min_consecutive = 40  # Подберите под задачу
    pred = predictions
    pred_defect_time = None
    count = 0
    for i in range(len(pred)):
        if pred[i] == 1:
            count += 1
            if count == min_consecutive:
                pred_defect_time = full_time[i - min_consecutive + 1]
                break
        else:
            count = 0

    # 3. Визуализация сравнения дефектов
    plt.figure(figsize=(14, 4))
    plt.plot(full_time, real_defect, label='Истинный дефект', drawstyle='steps-post', color='green', linewidth=2)
    plt.plot(full_time, pred, label='Предсказанный дефект', drawstyle='steps-post', color='red', alpha=0.7, linewidth=2)

    # Вертикальные линии и подписи
    if real_defect_time is not None:
        plt.axvline(real_defect_time, color='green', linestyle='--', linewidth=2, label='Реальный дефект')
        plt.annotate(f'Реальный дефект\n{real_defect_time:.2f} c',
                     xy=(real_defect_time, 1), xycoords='data',
                     xytext=(-60, 25), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='green'),
                     fontsize=11, color='green', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.7))
    if pred_defect_time is not None:
        plt.axvline(pred_defect_time, color='red', linestyle='--', linewidth=2, label='Предсказанный дефект')
        plt.annotate(f'Предсказанный дефект\n{pred_defect_time:.2f} c',
                     xy=(pred_defect_time, 1), xycoords='data',
                     xytext=(20, 25), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='red'),
                     fontsize=11, color='red', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.7))
        if real_defect_time is not None:
            time_error = pred_defect_time - real_defect_time
            plt.text(pred_defect_time, 1.12, f'Ошибка: {time_error:.2f} c', color='red', fontsize=12, va='bottom', ha='center')
    else:
        plt.text(full_time[-1], 1.12, 'Дефект не обнаружен моделью', color='red', fontsize=12, va='bottom', ha='right')

    # Метрики
    from sklearn.metrics import accuracy_score, roc_curve, auc
    acc = accuracy_score(real_defect, pred)
    fpr, tpr, _ = roc_curve(real_defect, proba)
    roc_auc = auc(fpr, tpr)
    plt.title(f"{file}\nAUC: {roc_auc:.3f}, Accuracy: {acc:.3f}", fontsize=14)
    plt.xlabel('Время')
    plt.ylabel('Метка дефекта')
    plt.ylim(-0.1, 1.25)
    plt.legend(loc='upper right')
    plt.tight_layout()
    compare_path = os.path.join(OUTPUT_FOLDER, f'compare_defect_{os.path.splitext(file)[0]}.png')
    plt.savefig(compare_path)
    plt.close()
    print(f"График сравнения меток дефекта сохранён: {compare_path}")

print("\nОбработка всех файлов завершена. Результаты в", OUTPUT_FOLDER)
