from datetime import timedelta
from pathlib import Path
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


with DAG(
    dag_id="employee_attrition",
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="PYTHONPATH=src python -m prediccion_fuga.data",
        cwd=str(PROJECT_ROOT)
    )

    train = BashOperator(
        task_id="train",
        bash_command="PYTHONPATH=src python -m prediccion_fuga.train",
        cwd=str(PROJECT_ROOT)
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python scripts/evaluate.py",
        cwd=str(PROJECT_ROOT)   
    )

    compare = BashOperator(
        task_id="compare",
        bash_command="python scripts/compare_models.py",
        cwd=str(PROJECT_ROOT)
    )

    promote = BashOperator(
        task_id="promote",
        bash_command="python scripts/promote_models.py",
        cwd=str(PROJECT_ROOT)   
    )

    prepare_data >> train >> evaluate >> compare >> promote
