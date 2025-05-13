from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import NewsDataPreprocessor

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'data_preprocessing_dag',
    default_args=default_args,
    description='DAG for preprocessing the news category dataset',
    schedule_interval=None,  # Manually triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nlp', 'preprocessing'],
)

# Create a preprocessor instance
preprocessor = NewsDataPreprocessor()

# Task 1: Check if raw dataset exists
check_raw_data = FileSensor(
    task_id='check_raw_data',
    filepath='/path/to/data/raw/news_category_dataset.csv',  # Update with actual path
    fs_conn_id='fs_default',
    poke_interval=60,
    timeout=60 * 5,
    mode='poke',
    dag=dag,
)

# Task 2: Basic cleaning
def basic_cleaning_task():
    df = preprocessor.load_data()
    df = preprocessor.basic_cleaning(df)
    return df.to_json()

basic_cleaning = PythonOperator(
    task_id='basic_cleaning',
    python_callable=basic_cleaning_task,
    dag=dag,
)

# Task 3: Advanced processing
def advanced_processing_task(**context):
    df_json = context['ti'].xcom_pull(task_ids='basic_cleaning')
    import pandas as pd
    df = pd.read_json(df_json)
    df = preprocessor.advanced_processing(df)
    return df.to_json()

advanced_processing = PythonOperator(
    task_id='advanced_processing',
    python_callable=advanced_processing_task,
    provide_context=True,
    dag=dag,
)

# Task 4: Feature engineering
def feature_engineering_task(**context):
    df_json = context['ti'].xcom_pull(task_ids='advanced_processing')
    import pandas as pd
    df = pd.read_json(df_json)
    df = preprocessor.feature_engineering(df)
    return df.to_json()

feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering_task,
    provide_context=True,
    dag=dag,
)

# Task 5: Save processed data
def save_processed_data_task(**context):
    df_json = context['ti'].xcom_pull(task_ids='feature_engineering')
    import pandas as pd
    df = pd.read_json(df_json)
    output_path = preprocessor.save_processed_data(df)
    return str(output_path)

save_processed_data = PythonOperator(
    task_id='save_processed_data',
    python_callable=save_processed_data_task,
    provide_context=True,
    dag=dag,
)

# Task 6: Log preprocessing completion
log_completion = BashOperator(
    task_id='log_completion',
    bash_command='echo "Data preprocessing completed successfully on $(date)"',
    dag=dag,
)

# Define task dependencies
check_raw_data >> basic_cleaning >> advanced_processing >> feature_engineering >> save_processed_data >> log_completion