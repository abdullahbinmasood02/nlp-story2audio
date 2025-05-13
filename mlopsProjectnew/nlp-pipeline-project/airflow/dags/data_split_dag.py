from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
import os
import sys

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.split import split_dataset

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'data_split_dag',
    default_args=default_args,
    description='DAG for splitting the processed dataset',
    schedule_interval=None,  # Manually triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nlp', 'data-split'],
)

# Task 1: Check if processed data exists
check_processed_data = FileSensor(
    task_id='check_processed_data',
    filepath='/path/to/data/processed/processed_news_data.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=60,
    timeout=60 * 5,
    mode='poke',
    dag=dag,
)

# Task 2: Split dataset
split_data_task = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset,
    op_kwargs={
        'test_size': 0.2,
        'valid_size': 0.1
    },
    dag=dag,
)

# Task 3: Check if split files were created
check_train_file = FileSensor(
    task_id='check_train_file',
    filepath='/path/to/data/processed/train.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=30,
    timeout=60 * 2,
    mode='poke',
    dag=dag,
)

check_valid_file = FileSensor(
    task_id='check_valid_file',
    filepath='/path/to/data/processed/validation.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=30,
    timeout=60 * 2,
    mode='poke',
    dag=dag,
)

check_test_file = FileSensor(
    task_id='check_test_file',
    filepath='/path/to/data/processed/test.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=30,
    timeout=60 * 2,
    mode='poke',
    dag=dag,
)

# Task 4: Notify completion
notify_completion = BashOperator(
    task_id='notify_completion',
    bash_command='echo "Data splitting completed successfully!"',
    dag=dag,
)

# Define task dependencies
check_processed_data >> split_data_task
split_data_task >> [check_train_file, check_valid_file, check_test_file]
[check_train_file, check_valid_file, check_test_file] >> notify_completion