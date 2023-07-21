import requests
import datetime
import json
import os
from dotenv import load_dotenv
from google.cloud.storage.client import Client as storage_client 
from google.cloud.bigquery.client import Client as bigquery_client 
from google.cloud.exceptions import Conflict
import pandas as pd
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

## Global Variables and Configs ##
load_dotenv("/opt/airflow/.env")
WORK_DIR = os.getenv("WORK_DIR") 
DATA_DIR = os.getenv("DATA_DIR")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'{WORK_DIR}/airflow/config/ServiceKey_GoogleCloud.json'
STORAGE_CLIENT = storage_client()
BIGQUERY_CLIENT = bigquery_client()
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d").replace("-","_")

## Weather Class ##
class WeatherPipline:

    def __init__(self, locations = None) -> None:
        self.locations = locations

    def extract_weather_data(self):
        '''
        Description: 
            - Extracts weather data from weatherapi.com
            - Stores data as txt files at {WORK_DIR}/data/{current_date}
        Args:
            None
        Returns:
            None 
        '''
        # Create a folder for the data
        if not os.path.exists(f"{WORK_DIR}/airflow/data/{CURRENT_DATE}"):
            os.mkdir(f"{WORK_DIR}/airflow/data/{CURRENT_DATE}")

        # Grab data for each location
        for location in self.locations:
            url = "https://weatherapi-com.p.rapidapi.com/current.json?q=" + location + "&lang=en"

            headers = {
                "X-RapidAPI-Key": "24cc538b51msh9dd38f0d1f4fd7ap150793jsn82c69f528d4e",
                "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers)

            print(response.json())

            # Generate text files for each location
            with open(f"{WORK_DIR}/airflow/data/{CURRENT_DATE}/{location}.txt", "w") as write_file:
                write_file.write(json.dumps(response.json()))
                write_file.close()

    def getOrCreate_bucket(self, bucket_name = f'weather_bucket_{CURRENT_DATE}'):
        '''
        Description:
            - Create bucket if doesnt exist
            - If exist, return it
        Args:
            bucket_name(str)
        Returns:
            bucket(Google Cloud Storage bucket)
        '''
        try:
            bucket = STORAGE_CLIENT.create_bucket(bucket_or_name=bucket_name)
        except Conflict:
            bucket = STORAGE_CLIENT.get_bucket(bucket_or_name=bucket_name)
        finally:
            return bucket

    def load_to_cloudStorage(self, bucket_name = f'weather_bucket_{CURRENT_DATE}', overwrite = False):
        '''
        Description:
            - getOrCreate a Cloud Storage bucket
            - Load today's text files
            - For cleanup, set overwrite = True
        Args:
            bucket_name(str)
            overwrite(bool)
        Returns:
            None
        '''

        # Delete bucket
        if overwrite:
            bucket = STORAGE_CLIENT.get_bucket(bucket_name)
            bucket.delete(force=True)

        # Load text files to bucket
        bucket = self.getOrCreate_bucket(bucket_name)
        os.chdir(f"{WORK_DIR}/airflow/data/{CURRENT_DATE}")
        for file in os.listdir():
            # load to tmp folder in bucket
            blob = bucket.blob(file + f"_{CURRENT_DATE}_{datetime.datetime.now().strftime('%H:%M:%S')}")
            blob.upload_from_filename(file)

    def process_data(self):
        '''
        Process raw weather data into a Pandas DataFrame
        Args:
            None
        Returns
            - df(pandas.DataFrame)
        '''
        # Change directory to today's data
        os.chdir(f"{DATA_DIR}/{CURRENT_DATE}")
        files = os.listdir()
        df = pd.DataFrame()
        current_index = 0

        # Read each file and append to a DataFrame
        for file in files:
            with open(file, 'r') as read_file:
                data = json.loads(read_file.read())

                # Extract data
                location_data = data.get("location")  
                current_data = data.get("current")

                # Create DataFrames
                location_df = pd.DataFrame(location_data, index=[current_index])
                current_df = pd.DataFrame(current_data, index=[current_index])
                current_index += 1
                current_df['condition'] = current_data.get('condition').get('text')

                # Concatenate DataFrames and append to main DataFrame
                temp_df = pd.concat([location_df, current_df],axis=1)
                df = pd.concat([df, temp_df])

                read_file.close()

        # Return main DataFrame
        df = df.rename(columns={'name':'city'})
        df['localtime'] = pd.to_datetime(df['localtime'])
        return df

    def getOrCreate_dataset(self, dataset_name :str = "weather"):
        '''
        Get dataset. If the dataset does not exists, create it.
        Args:
            - dataset_name(str) = Name of the new/existing data set.
            - project_id(str) = project id(default = The project id of the bigquery_client object)
        Returns:
            - dataset(google.cloud.bigquery.dataset.Dataset) = Google BigQuery Dataset
        '''
        print('Fetching Dataset...')
        try:
            # get and return dataset if exist
            dataset = BIGQUERY_CLIENT.get_dataset(dataset_name)
            print('Done')
            print(dataset.self_link)
            return dataset

        except Exception as e:
            # If not, create and return dataset
            if e.code == 404:
                print('Dataset does not exist. Creating a new one.')
                BIGQUERY_CLIENT.create_dataset(dataset_name)
                dataset = BIGQUERY_CLIENT.get_dataset(dataset_name)
                print('Done')
                print(dataset.self_link)
                return dataset
            else:
                print(e)


    def getOrCreate_table(self, dataset_name:str = "weather", table_name:str = "weather"):
        '''
        Create a table. If the table already exists, return it.
        Args:
            - table_name(str) = Name of the new/existing table.
            - dataset_name(str) = Name of the new/existing data set.
            - project_id(str) = project id(default = The project id of the bigquery_client object)
        Returns:
            - table(google.cloud.bigquery.table.Table) = Google BigQuery table
        '''
        # Grab prerequisites for creating a table
        dataset = self.getOrCreate_dataset()
        project = dataset.project
        dataset = dataset.dataset_id
        table_id = project + '.' + dataset + '.' + table_name

        print('\nFetching Table...')

        try:
            # Get table if exists
            table = BIGQUERY_CLIENT.get_table(table_id)
            print('Done')
            print(table.self_link)
        except Exception as e:
            # If not, create and get table
            if e.code == 404:
                print('Table does not exist. Creating a new one.')
                BIGQUERY_CLIENT.create_table(table_id)
                table = BIGQUERY_CLIENT.get_table(table_id)
                print(table.self_link)
            else:
                print(e)
        finally:
            return table

    def load_to_bigquery(self, dataframe, dataset_name, table_name):
        '''
        Description:
            - Get or Create a dataset
            - Get or Create a table
            - Load data from a DataFrame to BigQuery
        Args:
            dataset_name(str)
            table_name(str)
        Returns:
            None
        '''
        table = self.getOrCreate_table(dataset_name=dataset_name, table_name=table_name)
        BIGQUERY_CLIENT.load_table_from_dataframe(dataframe=dataframe, destination=table)

    def train_model(self, dataset_name, table_name):
        '''
        Description:
            - Get or Create a dataset
            - Get or Create a table
            - Load data from BigQuery to a DataFrame
            - Train a model
        Args:
            dataset_name(str)
            table_name(str)
        Returns:
            model(XGBRegressor)
        '''
        
        # Grab all the data from bigquery
        table = self.getOrCreate_table(dataset_name=dataset_name, table_name=table_name)
        query = f"select * from {table}"
        df = BIGQUERY_CLIENT.query(query).to_dataframe()

        # Pre processing
        df = df.drop(columns = ['region', 'country', 'tz_id', 'localtime','last_updated_epoch', 'last_updated', 'wind_dir', 'condition'])
        city_map = {
                'London':0,
                'Moscow':1,
                'Berlin':2,
                'Paris':3,
                'Rome':4,
                'Madrid':5,
                'Cairo':6,
                'Tokyo':7,
                'Sydney':8}
        df['city'] = df['city'].map(city_map)

        # divide to train test data
        x = df.drop(columns = ['temp_c'])
        y = df['temp_c']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True)

        # predict next day weather
        # Train an XGB model
        model = XGBRegressor()
        model.fit(x_train, y_train)

        # Predict test data and generate accuracy score
        predictions = model.predict(x_test)
        model_score = model.score(x_test, y_test)

        # Generate and print a predictions dataframe
        cities = []
        for city_number in x_test.city.to_list():
            for city, num in city_map.items():
                if city_number == num:
                    cities.append(city)
        predictions_df = pd.DataFrame([*zip(cities, y_test, predictions, abs(y_test-predictions), [model_score]*len(cities))], columns=['city', 'actual_temp(Celcius)', 'predicted_temp(Celcius)', 'diff(Celcius)','score'])
        print(f"Test Data Predictions:\n {predictions_df}")
        
        # Return trained model
        return model
    
    def predict_next_day_weather(self, model, dataset_name, table_name):
        '''
        Description:
            - Use an already-trained model to predict new data
        Args:
            model(XGBRegressor)
            dataset_name(str)
            table_name(str)
        Returns:
            predictions_df(pandas.DataFrame)
        '''
        # Set variables
        cities = ['London', 'Moscow' ,'Berlin' ,'Paris' ,'Rome' ,'Madrid', 'Cairo' ,'Tokyo', 'Sydney']
        next_day = datetime.datetime.now() + datetime.timedelta(days=1)
        next_day = next_day.strftime("%Y-%m-%d")
        table = self.getOrCreate_table(dataset_name=dataset_name, table_name=table_name)

        # Query BigQuery for the latest weather data
        query = f"""WITH RankedWeather AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY city ORDER BY localtime DESC) AS rn
                    FROM
                        {table}
                    )
                    SELECT
                    *
                    FROM
                    RankedWeather
                    WHERE
                    rn = 1;"""
        new_data = BIGQUERY_CLIENT.query(query).to_dataframe()

        # Pre processing
        new_data = new_data.drop(columns = ['temp_c','rn', 'region', 'country', 'tz_id', 'localtime','last_updated_epoch', 'last_updated', 'wind_dir', 'condition'])
        city_map = {
                'London':0,
                'Moscow':1,
                'Berlin':2,
                'Paris':3,
                'Rome':4,
                'Madrid':5,
                'Cairo':6,
                'Tokyo':7,
                'Sydney':8}
        new_data['city'] = new_data['city'].map(city_map)

        # Add 24 hours to date related columns
        new_data['localtime_epoch'] = new_data['localtime_epoch'] + 86400

        # Predict next day weather
        predictions = model.predict(new_data)

        # Generate and print a predictions dataframe
        predictions_df = pd.DataFrame([*zip(cities, predictions)], columns=['city', 'predicted_temp(Celcius)'])
        predictions_df['at_date(UTC+0)'] = new_data['localtime_epoch']

        # translate epoch to datetime
        predictions_df['at_date(UTC+0)'] = pd.to_datetime(predictions_df['at_date(UTC+0)'], unit='s')
        print(f"Next Day Predictions:\n {predictions_df}")

        return predictions_df
   


if __name__ == "__main__":
    locations = ["London", "Tokyo", "Sydney", "Paris", "Berlin", "Moscow", "Madrid", "Rome", "Cairo"]
    weather = WeatherPipline(locations)
    #weather.extract_weather_data()
    #weather.load_to_cloudStorage()
    #df = weather.process_data()
    #weather.load_to_bigquery(dataframe = df, dataset_name='weather', table_name='weather')
    model = weather.train_model(dataset_name='weather', table_name='weather')
    predictions_df = weather.predict_next_day_weather(model, dataset_name='weather', table_name='weather')
    
