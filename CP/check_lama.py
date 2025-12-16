import pandas as pd
from sklearn.model_selection import train_test_split

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML

DATA_PATH = r"C:\Users\tsura\PycharmProjects\PNI_PC\train_with_mean_amplitude_and_defect\U_PD4_with_defect_mean_amplitude_defect.csv"
TARGET_COL = "Defect"

df = pd.read_csv(DATA_PATH)

train, valid = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_COL]
)

roles = {"target": TARGET_COL}

# ЯВНО создаём объект задачи: бинарная классификация
task = Task("binary")

automl = TabularAutoML(task=task)
oof = automl.fit_predict(train, roles=roles, valid_data=valid)

print("OK, LightAutoML отработал, shape OOF:", oof.shape)
