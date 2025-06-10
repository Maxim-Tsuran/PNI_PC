import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# здесь происходит обработка данных, определяются и сохраняются значения амплитудные по окнам, чтобы потом их подавать на новую модель CatBoost
# Так же здесь обрабатываются данные для проверки работы модели катбуст и загружаются в папку test_with_mean_amplitude_and_defect

# Дефект при амплитуде 0.0003, задаем вручную уставки по времени около этого значения


# === ПАРАМЕТРЫ ===
window_size = 15000

def moving_average_abs(y, window):
    return np.convolve(np.abs(y), np.ones(window)/window, mode='valid')

# === ОТДЕЛЬНЫЕ УСТАВКИ ДЛЯ TRAIN И TEST ===
train_defect_time_thresholds = {
    'U_PD4_with_defect': 0.69,
    'U_PD5_with_defect': 1.535,
    'U_PD6_with_defect': 4.76,
    'U_PD_with_defect': 1.17,
}

test_defect_time_thresholds = {
    'U_PD4_with_defect': 0.8,
    'U_PD5_with_defect': 1.5,
    'U_PD6_with_defect': 4.8,
    'U_PD_with_defect': 1.01,
}

# === ОБРАБОТКА ОБУЧАЮЩИХ ДАННЫХ ===
input_folder = 'train_with_defect'
output_folder = 'train_with_mean_amplitude_and_defect'
plot_folder = 'train_with_mean_amplitude_plots'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

files = glob.glob(f'{input_folder}/*.csv')

for file in files:
    df = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]

    if not {'Time', 'PD_Level', 'Defect'}.issubset(df.columns):
        print(f'Пропущены нужные столбцы в {file}')
        continue

    ma_amplitude = moving_average_abs(df['PD_Level'].values, window_size)
    time = df['Time'].values
    pd_level = df['PD_Level'].values
    if window_size % 2 == 1:
        ma_time = time[window_size//2:-(window_size//2)]
        pd_level_center = pd_level[window_size//2:-(window_size//2)]
    else:
        ma_time = time[window_size//2-1:-(window_size//2)]
        pd_level_center = pd_level[window_size//2-1:-(window_size//2)]

    threshold = train_defect_time_thresholds.get(base_name, None)
    if threshold is not None:
        defect = (ma_time >= threshold).astype(int)
    else:
        print(f'Уставка для файла {base_name} не задана, метка Defect будет 0')
        defect = np.zeros_like(ma_time, dtype=int)

    df_new = pd.DataFrame({
        'Time': ma_time,
        'PD_Level': pd_level_center,
        'Mean_Amplitude': ma_amplitude,
        'Defect': defect
    })

    output_file = os.path.join(output_folder, f'{base_name}_mean_amplitude_defect.csv')
    df_new.to_csv(output_file, index=False)
    print(f'Сохранён файл: {output_file}')

    # === График с выделением зоны дефекта ===
    plt.figure(figsize=(10, 5))
    plt.plot(ma_time, ma_amplitude, label='Mean Amplitude', color='blue')
    plt.fill_between(ma_time, 0, ma_amplitude, where=defect==1, color='red', alpha=0.2, label='Defect=1')
    plt.xlabel('Time')
    plt.ylabel('Mean Amplitude')
    plt.title(f'Скользящее среднее амплитуды: {base_name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(plot_folder, f'{base_name}_mean_amplitude.png')
    plt.savefig(plot_file, dpi=200)
    # plt.show()
    plt.close()
    print(f'График сохранён: {plot_file}')

print('Обработка обучающих файлов завершена.\n')

# === ОБРАБОТКА ТЕСТОВЫХ ДАННЫХ ===
test_input_folder = 'test_with_defect'
test_output_folder = 'test_with_mean_amplitude_and_defect'
test_plot_folder = 'test_with_mean_amplitude_plots'
os.makedirs(test_output_folder, exist_ok=True)
os.makedirs(test_plot_folder, exist_ok=True)

test_files = glob.glob(f'{test_input_folder}/*.csv')

for file in test_files:
    df = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]

    if not {'Time', 'PD_Level', 'Defect'}.issubset(df.columns):
        print(f'Пропущены нужные столбцы в {file}')
        continue

    ma_amplitude = moving_average_abs(df['PD_Level'].values, window_size)
    time = df['Time'].values
    pd_level = df['PD_Level'].values
    if window_size % 2 == 1:
        ma_time = time[window_size//2:-(window_size//2)]
        pd_level_center = pd_level[window_size//2:-(window_size//2)]
    else:
        ma_time = time[window_size//2-1:-(window_size//2)]
        pd_level_center = pd_level[window_size//2-1:-(window_size//2)]

    threshold = test_defect_time_thresholds.get(base_name, None)
    if threshold is not None:
        defect = (ma_time >= threshold).astype(int)
    else:
        print(f'Уставка для файла {base_name} не задана, метка Defect будет 0')
        defect = np.zeros_like(ma_time, dtype=int)

    df_new = pd.DataFrame({
        'Time': ma_time,
        'PD_Level': pd_level_center,
        'Mean_Amplitude': ma_amplitude,
        'Defect': defect
    })

    output_file = os.path.join(test_output_folder, f'{base_name}_mean_amplitude_defect.csv')
    df_new.to_csv(output_file, index=False)
    print(f'Сохранён ТЕСТОВЫЙ файл: {output_file}')

    # === График с выделением зоны дефекта ===
    plt.figure(figsize=(10, 5))
    plt.plot(ma_time, ma_amplitude, label='Mean Amplitude', color='blue')
    plt.fill_between(ma_time, 0, ma_amplitude, where=defect==1, color='red', alpha=0.2, label='Defect=1')
    plt.xlabel('Time')
    plt.ylabel('Mean Amplitude')
    plt.title(f'Скользящее среднее амплитуды (тест): {base_name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(test_plot_folder, f'{base_name}_mean_amplitude.png')
    plt.savefig(plot_file, dpi=200)
    plt.show()
    plt.close()
    print(f'График (тест) сохранён: {plot_file}')

print('Обработка тестовых файлов завершена.')
