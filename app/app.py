from flask import Flask
import requests
from bs4 import BeautifulSoup
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
WORK_DIR = os.getenv("WORK_DIR")
sys.path.append(f"{WORK_DIR}/airflow")
from helper.weather import WeatherPipline
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'{WORK_DIR}/airflow/config/ServiceKey_GoogleCloud.json'

# Creating Flask app
app = Flask(__name__)

# Creating WeatherPipline object
locations = ["London", "Tokyo", "Sydney", "Paris", "Berlin", "Moscow", "Madrid", "Rome", "Cairo"]
weather = WeatherPipline(locations)

@app.route("/city=<city>")
def get_data(city):

    # Train model and predict next day weather
    model = weather.train_model('weather', 'weather')
    predictions_df = weather.predict_next_day_weather(model, 'weather', 'weather')
    predictions_df['at_date(UTC+0)'] = predictions_df['at_date(UTC+0)'].astype(str)

    # Return all predictions or predictions for a specific city
    if city == 'All':
        return predictions_df.to_json(orient='records')
    else:
        return predictions_df[predictions_df['city'] == city].to_json(orient='records')

if __name__ == "__main__":
    app.run(debug=True)

