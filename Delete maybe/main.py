#  Импортируем нужные библиотеки
import pandas as pd                      # Работа с таблицами
import numpy as np                       # Работа с числами и массивами
import glob                              # Поиск файлов по шаблону, например "*.csv"
import matplotlib.pyplot as plt          # Рисование графиков
from catboost import CatBoostClassifier  # модель машинного обучения
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score  # Метрики оценки качества модели
import joblib                            # Сохранение модели на диск
import os

# Старое, не смотреть



# === 1. ЗАГРУЖАЕМ ОБУЧАЮЩИЕ ДАННЫЕ ===

# Ищем все CSV-файлы в папке "train_with_defect"
train_files = glob.glob("train_with_defect/*.csv")

# Читаем каждый CSV-файл в виде таблицы и складываем в список
df_list = [pd.read_csv(file) for file in train_files]

# Объединяем все таблицы в одну большую таблицу (как будто склеиваем их по строкам)
train_df = pd.concat(df_list, ignore_index=True)

# === 2. ОБУЧАЕМ МОДЕЛЬ ===

# Выбираем признаки (что будет "входом" для модели)
X_train = train_df[['Time', 'PD_Level']]  # Время и уровень сигнала

# Берем метку (что модель должна предсказывать — есть дефект или нет)
y_train = train_df['Defect']

# Создаём модель машинного обучения
model = CatBoostClassifier(verbose=0)  # verbose=0 — чтобы не показывал прогресс в консоли

# Обучаем модель на наших данных
model.fit(X_train, y_train)

# === 3. ОЦЕНИВАЕМ КАЧЕСТВО МОДЕЛИ ===

# Предсказываем, что думает модель на тех же данных
y_pred = model.predict(X_train)               # Классы: 0 или 1
y_proba = model.predict_proba(X_train)[:, 1]  # Вероятности дефекта (0.0–1.0)

# Точность — сколько предсказаний совпало с реальностью
print("Accuracy:", accuracy_score(y_train, y_pred))

# Матрица ошибок — показывает, сколько было ошибок и где
cm = confusion_matrix(y_train, y_pred)
print("Confusion matrix:\n", cm)

# Строим ROC-кривую — показывает, насколько хорошо модель отделяет дефект от нормального состояния
fpr, tpr, _ = roc_curve(y_train, y_proba)
roc_auc = auc(fpr, tpr)  # Площадь под ROC-кривой — чем ближе к 1.0, тем лучше

# Рисуем график ROC-кривой
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Линия случайного угадывания
plt.xlabel('False Positive Rate')  # Ошибочные срабатывания
plt.ylabel('True Positive Rate')   # Настоящие срабатывания
plt.title('ROC Curve')             # Заголовок графика
plt.legend()                       # Обозначения
plt.grid()                         # Сетка
plt.savefig('1')                   # Сохраняем график в файл "1.png"

# === 4. СОХРАНЯЕМ ОБУЧЕННУЮ МОДЕЛЬ ===

joblib.dump(model, '../catboost_defect_predictor.cbm')  # Сохраняем модель в файл
print("Модель сохранена: catboost_defect_predictor.cbm")

# === 5. ПРЕДСКАЗЫВАЕМ НА НОВЫХ ДАННЫХ (ГДЕ НЕТ МЕТОК) ===

# Ищем все CSV-файлы в папке с новыми данными
predict_files = glob.glob("test_with_defect/*.csv")

# Папка для сохранения результатов (может быть любой)
output_folder = "predicted"

# Для каждого файла:
for file in predict_files:
    df = pd.read_csv(file)                        # Читаем файл
    X_new = df[['Time', 'PD_Level']]              # Берем те же признаки
    predictions = model.predict(X_new)            # Предсказываем, где дефект (0 или 1)
    proba = model.predict_proba(X_new)[:, 1]      # Предсказываем вероятность дефекта

    df['Predicted_Defect'] = predictions          # Добавляем предсказания в таблицу
    df['Defect_Probability'] = proba              # Добавляем вероятность

    # Формируем новый путь: "папка/имя_файла"
    file_name = os.path.basename(file)            # Получаем только имя файла (без пути)
    output_name = os.path.join(output_folder, file_name.replace(".csv", "_predicted.csv"))

    # Сохраняем в новую папку
    df.to_csv(output_name, index=False)
    print(f"Предсказания сохранены в: {output_name}")

    # === РИСУЕМ ГРАФИК ===
    plt.figure(figsize=(8, 3))
    plt.plot(df['Time'], df['PD_Level'], label='PD Level')  # Сигнал
    plt.plot(df['Time'], df['Defect_Probability'], label='Defect Probability', color='red')  # Вероятность дефекта
    plt.fill_between(df['Time'], 0, 1,
                     where=df['Predicted_Defect'] == 1,
                     color='orange', alpha=0.2,
                     label='Defect Zone')  # Закрашиваем зоны, где модель видит дефект

    plt.title(f'Defect Prediction: {file}')  # Название графика
    plt.xlabel('Time')
    plt.ylabel('Level / Probability')
    plt.legend(loc='upper right')  # Убираем warning, фиксируем легенду в правом верхнем углу
    plt.tight_layout()
    i=+1
    plt.savefig('plot№_' + str(i) + '.png')  # Сохраняем картинку как "2.png"
