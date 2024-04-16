from helper_functions import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from tensorflow.keras.models import Sequential 

def part_3():
    #load the training data
    #train_data = get_train_data()

    #Only need windspeed data (10 meters above ground level)
    train_data = pd.read_csv("TrainData.csv")
    train_data['TIMESTAMP'] = pd.to_datetime(train_data['TIMESTAMP'], format="%Y%m%d %H:%M")
    train_data['TIMESTAMP'] = train_data['TIMESTAMP'].astype(int) / 10**9
    train_data['TIMESTAMP'] = train_data['TIMESTAMP'].astype(int)
    train_X = train_data[['TIMESTAMP']]
    train_Y = train_data.copy()
    #train_Y.set_index('TIMESTAMP', inplace=True)
    train_Y = train_Y[['POWER']]

    #load the test data
    test_X = pd.read_csv("WeatherForecastInput.csv")
    test_X = test_X[['TIMESTAMP']]
    test_X_COPY = test_X.copy()
    test_X['TIMESTAMP'] = pd.to_datetime(test_X['TIMESTAMP'], format="%Y%m%d %H:%M")
    test_X['TIMESTAMP'] = test_X['TIMESTAMP'].astype(int) / 10**9
    test_X['TIMESTAMP'] = test_X['TIMESTAMP'].astype(int)
    test_Y = get_test_data_Y()

    #1. Linear regression
    #NOTE: fit_intercept=True by default. If the data is centered, set to False
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_X, train_Y)

    predictions = lin_reg_model.predict(test_X)
    rmse_lin_reg = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X_COPY['TIMESTAMP'], columns=['POWER']).to_csv("out/part_3/ForecastTemplate3-LR.csv")

    #2. SVR regression
    #TODO: Tune hyperparameters
    svr_model = SVR()
    svr_model.fit(train_X, train_Y)

    predictions = svr_model.predict(test_X)
    rmse_SVG = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X_COPY['TIMESTAMP'], columns=['POWER']).to_csv("out/part_3/ForecastTemplate3-SVR.csv")

    #3. ANN regression
    #TODO: Tune hyperparameters
    neural_network = MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=1000, activation='relu')
    neural_network.fit(train_X, train_Y)

    predictions = neural_network.predict(test_X)
    rmse_NN = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X_COPY['TIMESTAMP'], columns=['POWER']).to_csv("out/part_3/ForecastTemplate3-NN.csv")


    #4. RNN regression
    simple_rnn = Sequential(
        [ #Setting up the layers
            Dense(1),
            LSTM(20, activation='relu', input_shape=(1, 1)),
            Dense(20, activation='relu'),
            Dense(1)
        ]

    )

    simple_rnn.compile(
        optimizer='adam', 
        loss='mean_squared_error',
    )

    #Reshape the data
    train_X = train_X.values.reshape(-1, 1, 1)
    
    history = simple_rnn.fit(
        train_X, train_Y,
        epochs=10,
        batch_size=32,
    )

    predictions = simple_rnn.predict(test_X, batch_size=32)
    rmse_RNN = root_mean_squared_error(test_Y, predictions)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X_COPY['TIMESTAMP'], columns=['POWER']).to_csv("out/part_3/ForecastTemplate3-RNN.csv")

    #5. Plot the RMSE values
    models = ['Linear Regression', 'SVR', 'Neural Network', 'RNN']
    rmse_values = [rmse_lin_reg, rmse_SVG, rmse_NN, rmse_RNN]

    #6. Table of RMSE values
    print("RMSE values for different models:")
    table_data = { 'Model': models, 'RMSE': rmse_values }
    pd_table = pd.DataFrame(table_data)
    print(pd_table)


if __name__ == "__main__":
    part_3()