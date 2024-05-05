import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from helper_functions import get_test_data_Y

def part_3():
    RND_SEED = 42
    WINDOW_SIZE = 1

    # Load test data
    test_data = get_test_data_Y()  # Assume this function returns a DataFrame with a 'POWER' column
    test_data = test_data['POWER'].values

    features_test, targets_test = [], []
    for i in range(WINDOW_SIZE, len(test_data)):
        features_test.append(test_data[i-WINDOW_SIZE:i])
        targets_test.append(test_data[i])

    test_X = pd.DataFrame(features_test, columns=[f'POWER_t-{i}' for i in range(WINDOW_SIZE, 0, -1)])
    test_Y = pd.DataFrame(targets_test, columns=['POWER'])
    test_X.index = pd.date_range(start='2013-11-01', periods=len(test_X), freq='H')  # November dates

    # Load training data
    data = pd.read_csv("TrainData.csv")
    power_data = data['POWER'].values

    features, targets = [], []
    for i in range(WINDOW_SIZE, len(power_data)):
        features.append(power_data[i-WINDOW_SIZE:i])
        targets.append(power_data[i])

    train_X = pd.DataFrame(features, columns=[f'POWER_t-{i}' for i in range(WINDOW_SIZE, 0, -1)])
    train_Y = pd.DataFrame(targets, columns=['POWER'])

    # Initialize models
    models = {
        'LR': LinearRegression(),
        'SVR': SVR(),
        'ANN': MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=1000, activation='relu', random_state=RND_SEED),
        'RNN': Sequential([SimpleRNN(20, activation='relu', input_shape=(WINDOW_SIZE, 1)), Dense(1)])
    }

    rmses = {}

    # Train models and make predictions
    for name, model in models.items():
        if name == 'RNN':
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(train_X.values.reshape(-1, WINDOW_SIZE, 1), train_Y, epochs=50, batch_size=30)
            predictions = model.predict(test_X.values.reshape(-1, WINDOW_SIZE, 1)).flatten()
        else:
            model.fit(train_X, train_Y.values.ravel())
            predictions = model.predict(test_X)

        # Save predictions to CSV
        save_predictions(test_X.index, predictions, f'ForecastTemplate3-{name}.csv')

        # Calculate RMSE and store it
        rmse = mean_squared_error(test_Y, predictions, squared=False)
        rmses[name] = rmse

    # Output all RMSEs with six decimal places
    print("RMSEs for all models:")
    for model, rmse in rmses.items():
        print(f"{model}: {rmse:.6f}")

    # Plotting
    plot_time_series(test_X.index, test_Y['POWER'], models['LR'].predict(test_X), models['SVR'].predict(test_X), 'LR & SVR')
    plot_time_series(test_X.index, test_Y['POWER'], models['ANN'].predict(test_X), models['RNN'].predict(test_X.values.reshape(-1, WINDOW_SIZE, 1)).flatten(), 'ANN & RNN')

def save_predictions(dates, predictions, filename):
    output_dir = "out/part_3/"
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.title(f'Wind Power Predictions for November 2013 - {title_suffix}')
    plt.xlabel('Date (dd-mm)')
    plt.ylabel('Wind Power')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    part_3()
