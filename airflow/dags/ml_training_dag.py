from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
}

#schedule="@daily"

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    schedule="*/2 * * * *",
    catchup=False,
) as dag:

    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command="""
        python /opt/airflow/dags/ingest_data.py
        """
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="""
        python /opt/airflow/dags/validate_data.py
        """
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
        python /opt/airflow/dags/train_model.py
        """
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="""
        python /opt/airflow/dags/evaluate_model.py
        """
    )

    ingest_data >> validate_data >> train_model >> evaluate_model