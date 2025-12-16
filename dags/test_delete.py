from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

with DAG(
    dag_id="test_simple",
    start_date=datetime(2025, 12, 1),
    schedule=None,
    catchup=False,
) as dag:
    EmptyOperator(task_id="ok")
