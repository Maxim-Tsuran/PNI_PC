Запуск Apache Airflow:
- Создать файлы .env, docker-compose.yaml, Dockerfile перед работой с содержимым, которое сейчас внутри.
1. Открыть CMD
2. Пройти по пути cd C:\Users\tsura\PycharmProjects\PNI_PC
3. docker compose down
4. docker compose build --no-cache   (под вопросом. Мб сбросит все записи, лучше пропустить этот шаг)
5. docker compose up -d   (Поднятие Apache Airflow с помощью докера)

Запуск MLFlow:
1. Открыть CMD
2. Пройти по пути cd C:\Users\tsura\PycharmProjects\PNI_PC
3. python -m mlflow ui --backend-store-uri "file:///C:/Users/tsura/PycharmProjects/PNI_PC/mlruns" --host 127.0.0.1 --port 5000
