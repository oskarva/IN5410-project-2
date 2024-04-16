from helper_functions import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def part_2():
    
    #load the training data
    train_data = get_train_data()
    
    #load the test data
    weather_forecast_input = get_test_data_X()
    solution = get_test_data_Y()

    #Preprocess the data 
    # Calculate wind direction from U10 and V10
    train_data['wind_direction'] = np.arctan2(train_data['V10'], train_data['U10']) * (180 / np.pi)
    weather_forecast_input['wind_direction'] = np.arctan2(weather_forecast_input['V10'], weather_forecast_input['U10']) * (180 / np.pi)
    
    # Define features (wind speed and wind direction) and target variable (wind power)
    X_train = train_data[['WS10', 'wind_direction']]
    y_train = train_data[['POWER']]

    # Implement and train the MLR model
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)

    #  Make predictions for the wind power production for November 2013
    X_test = weather_forecast_input[['WS10', 'wind_direction']]
    wind_power_predictions = mlr_model.predict(X_test)
    
    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(wind_power_predictions, index=X_test.index, columns=['POWER']).to_csv("out/part_2/ForecastTemplate2.csv")

    # Evaluate the predictions using RMSE
    true_wind_power = solution[['POWER']]
    rmse = root_mean_squared_error(true_wind_power, wind_power_predictions)
    print("RMSE:", rmse)

    # Compare the prediction accuracy with the linear regression model using only wind speed
    # Define features (wind speed) and target variable (wind power)
    # Maybe should integrate what has been done in part one instead of redoing it?
    X_train_ws = train_data[['WS10']]
    X_test_ws = weather_forecast_input[['WS10']]

    # Implement and train the linear regression model using only wind speed
    lr_model_ws = LinearRegression()
    lr_model_ws.fit(X_train_ws, y_train)

    # Make predictions for wind power using linear regression with only wind speed
    wind_power_predictions_ws = lr_model_ws.predict(X_test_ws)

    # Evaluate the predictions using RMSE
    rmse_ws = root_mean_squared_error(true_wind_power, wind_power_predictions_ws)
    print("RMSE (Linear Regression with Wind Speed only):", rmse_ws)
    
    
    #PLOTS    
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(solution.index), solution[['POWER']], label='True Wind Energy Measurement')
    plt.plot(pd.to_datetime(weather_forecast_input.index), wind_power_predictions_ws, label='Linear Regression')
    plt.plot(pd.to_datetime(weather_forecast_input.index), wind_power_predictions, label='Multiple Linear Regression')
    plt.xlabel('Date (dd-mm)')
    plt.ylabel('Wind Power')
    # plt.title('Wind Power Forecasting for November 2013')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    # Set x-axis ticks to be at the beginning of each day
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.savefig("out/part_2/part_2_plot.eps")
    plt.show()


    # Calculate RMSE and compare prediction accuracy
    rmse_table = pd.DataFrame({'Model': ['Linear Regression', 'Multiple Linear Regression'],
                            'RMSE': [rmse_ws, rmse]})
    print(rmse_table)
    return None

if __name__ == "__main__":
    part_2()