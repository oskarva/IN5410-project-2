from helper_functions import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    part_3()