from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import json

app = Flask(__name__)

# Load or generate sample data (replace this with actual data)
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
    'energy_usage': [20 + 5 * i for i in range(100)],
    'temperature': [25 + 2 * i for i in range(100)],
    'day_of_week': [i % 7 for i in range(100)],
    'hour_of_day': [i % 24 for i in range(100)],
    'month': [i % 12 + 1 for i in range(100)],
    'weather_condition': ['sunny', 'cloudy', 'rainy'] * 33+['sunny']  # Example weather data
}
df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

# Train the model
model = ExponentialSmoothing(df['energy_usage'], trend='add', seasonal='add', seasonal_periods=24)
model_fit = model.fit()

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        input_data = request.get_json()

        current_energy_consumption = input_data['current_energy_consumption']
        current_temperature = input_data['current_temperature']
        current_day_of_week = input_data['current_day_of_week']
        current_hour_of_day = input_data['current_hour_of_day']
        current_month = input_data['current_month']
        current_weather_condition = input_data['current_weather_condition']

        # Forecast energy consumption
        forecast_steps = input_data.get('forecast_steps', 1)
        forecasts = model_fit.forecast(steps=forecast_steps)

        result = {
            'current_energy_consumption': current_energy_consumption,
            'current_temperature': current_temperature,
            'current_day_of_week': current_day_of_week,
            'current_hour_of_day': current_hour_of_day,
            'current_month': current_month,
            'current_weather_condition': current_weather_condition,
            'forecasts': forecasts.tolist()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
