from helper_functions import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, RNN
from tensorflow.keras.models import Sequential 
import numpy as np
import os
import matplotlib.dates as mdates


def part_3():
    #constants
    RND_SEED = 42
    WINDOW_SIZE = 1

    #load the test data
    test_data = get_test_data_Y() #Y, because we are only interested in power data, and WeatherForecastInput.csv does not contain power data
    test_data = test_data['POWER'].values

    features_test, targets_test = [], []

    for i in range(WINDOW_SIZE, len(test_data)):
        features_test.append(test_data[i-WINDOW_SIZE:i])
        targets_test.append(test_data[i])
    
    # Convert to numpy arrays for use in training
    test_X = pd.DataFrame(features_test, columns=[f'POWER_t-{i}' for i in range(WINDOW_SIZE, 0, -1)])
    test_Y = pd.DataFrame(targets_test, columns=[f'POWER'])
    test_X.index = pd.date_range(start='2013-11-01', periods=len(test_X), freq='H')  # November dates



    #Load training data
    data = pd.read_csv("TrainData.csv")
    power_data = data['POWER'].values

    #Prepare features and targets
    features, targets = [], []

    for i in range(WINDOW_SIZE, len(power_data)):
        features.append(power_data[i-WINDOW_SIZE:i])
        targets.append(power_data[i])

    # Convert to numpy arrays for use in training
    train_X = pd.DataFrame(features, columns=[f'POWER_t-{i}' for i in range(WINDOW_SIZE, 0, -1)])
    train_Y = pd.DataFrame(targets, columns=[f'POWER'])


    #1. Linear regression
    #NOTE: fit_intercept=True by default. If the data is centered, set to False
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_X, train_Y)

    predictions = lin_reg_model.predict(test_X)
    rmse_lin_reg = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    save_predictions(test_X.index, predictions, 'ForecastTemplate3-LR.csv')

    #2. SVR regression
    svr_model = SVR()
    svr_model.fit(train_X, train_Y)

    predictions = svr_model.predict(test_X)
    rmse_SVG = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    save_predictions(test_X.index, predictions, 'ForecastTemplate3-SVR.csv')

    #3. ANN regression
    neural_network = MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=1000, activation='relu', random_state=RND_SEED)
    neural_network.fit(train_X, train_Y)

    predictions = neural_network.predict(test_X)
    rmse_NN = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    save_predictions(test_X.index, predictions, 'ForecastTemplate3-ANN.csv')


    #4. RNN regression
    simple_rnn = Sequential(
        [ #Setting up the layers
            SimpleRNN(20, activation='relu', input_shape=(WINDOW_SIZE, 1)), #RNN layer
            Dense(1), #output layer
        ]

    )

    simple_rnn.compile(
        optimizer='adam', 
        loss='mean_squared_error',
    )

    #Reshape the data
    train_X = train_X.values.reshape(-1, WINDOW_SIZE, 1)
    
    history = simple_rnn.fit(
        train_X, train_Y,
        epochs=50, 
        batch_size=30, 
    )

    predictions = simple_rnn.predict(test_X, batch_size=1)
    rmse_RNN = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    save_predictions(test_X.index, predictions, 'ForecastTemplate3-RNN.csv')

    #5. Plot the RMSE values
    models = ['Linear Regression', 'SVR', 'Neural Network', 'RNN']
    rmse_values = [rmse_lin_reg, rmse_SVG, rmse_NN, rmse_RNN]

    #6. Table of RMSE values
    print("RMSE values for different models:")
    table_data = { 'Model': models, 'RMSE': rmse_values }
    pd_table = pd.DataFrame(table_data)
    print(pd_table)

    # Plotting
    plot_time_series(test_X.index, test_Y['POWER'], lin_reg_model.predict(test_X), svr_model.predict(test_X), 'LR & SVR')
    plot_time_series(test_X.index, test_Y['POWER'], neural_network.predict(test_X), simple_rnn.predict(test_X, batch_size=1).flatten(), 'ANN & RNN')



def save_predictions(dates, predictions, filename):
    output_dir = "out/part_3/"
    os.makedirs(output_dir, exist_ok=True)

    #ensure predictions are flattened to a 1-dimensional array
    predictions = np.array(predictions).flatten()
    
    forecast_df = pd.DataFrame({
        'TIMESTAMP': pd.to_datetime(dates),
        'FORECAST': predictions
    })
    forecast_df.to_csv(output_dir + filename, index=False)

def plot_time_series(dates, real_data, predictions1, predictions2, title_suffix):
    dates = pd.to_datetime(dates)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, real_data, label='Real Wind Power', color='blue', linestyle='-')
    plt.plot(dates, predictions1, label=f'{title_suffix.split(" & ")[0]} Predictions', color='red', linestyle='-')
    plt.plot(dates, predictions2, label=f'{title_suffix.split(" & ")[1]} Predictions', color='green', linestyle='-')
    plt.xlabel('Date (dd-mm)')
    plt.ylabel('Wind Power')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'out/part_3/{title_suffix}.png')
    plt.show()

if __name__ == "__main__":
    part_3()