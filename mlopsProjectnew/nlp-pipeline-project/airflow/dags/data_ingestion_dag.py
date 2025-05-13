from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
import os
import sys

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.download import download_dataset

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG for ingesting news category dataset',
    schedule_interval=timedelta(days=7),  # Run weekly
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nlp', 'ingestion'],
)

# Task 1: Download dataset
download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag,
)

# Task 2: File sensor to check if dataset exists
file_sensor = FileSensor(
    task_id='check_dataset_file',
    filepath='/path/to/data/raw/news_category_dataset.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=60,  # Check every minute
    timeout=60 * 5,  # Timeout after 5 minutes
    mode='poke',
    dag=dag,
)

# Task 3: Log dataset information
log_info_task = BashOperator(
    task_id='log_dataset_info',
    bash_command='echo "Dataset ingestion completed successfully on $(date)"',
    dag=dag,
)

# Define task dependencies
download_task >> file_sensor >> log_info_task