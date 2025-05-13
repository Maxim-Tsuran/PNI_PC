import pandas as pd
import matplotlib.pyplot as plt
import os

# Графики для тестовых данных
project_folder = os.getcwd()
folder_path = 'test_with_defect'  # Укажите путь к вашей папке
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
output_folder = os.path.join(project_folder, 'row_plots.png')

for csv_file in csv_files:
    # Читаем CSV
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)

    # Проверяем структуру данных (первый столбец - время, второй - значение)
    time = data.iloc[:, 0]  # Первый столбец
    values = data.iloc[:, 1]  # Второй столбец

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(time, values, label=csv_file)

    # Настраиваем подписи
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title(f'График из файла {csv_file}')
    plt.legend()
    plt.grid(True)

    # Сохраняем в папку 'plots'
    output_path = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

