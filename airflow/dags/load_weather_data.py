from airflow import DAG
from airflow.decorators import task
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import sys
load_dotenv()
WORK_DIR = os.environ["WORK_DIR"]
sys.path.append(f"{WORK_DIR}/airflow")
from helper.weather import WeatherPipline

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'{WORK_DIR}/config/ServiceKey_GoogleCloud.json'

locations = ["London", "Tokyo", "Sydney", "Paris", "Berlin", "Moscow", "Madrid", "Rome", "Cairo"]
weather = WeatherPipline(locations)

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id = 'load_weather_data',
    default_args=default_args, 
    start_date = datetime(2023,7,20), 
    schedule_interval='@hourly', 
    catchup=False
) as dag:

    # Task #1 - Extract data
    @task
    def extract_weather_data():
        weather.extract_weather_data()

    # Task #2 - load_to_cloudStorage
    @task
    def load_to_cloudStorage():
        weather.load_to_cloudStorage()

    # Task #3 - load_to_bigquery
    @task
    def load_to_bigquery(dataset_name, table_name):
        df = weather.process_data()
        weather.load_to_bigquery(df, dataset_name, table_name)


    # Dependencies
    extract_weather_data() >> load_to_cloudStorage() >> load_to_bigquery("weather", "weather")