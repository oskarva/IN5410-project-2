from helper_functions import *
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#Training data = TrainData.csv, ONLY 10m above ground level
#relationship between generation and wind speed
#Test data = WeatherForecastInput.csv (test inputs) and Solution.csv (gold standard)
#Generates: four output .csv files 
def part_1():

    #load the training data
    train_data = get_train_data()

    #Only need windspeed data (10 meters above ground level)
    train_X = train_data[['WS10']]
    train_Y = train_data[['POWER']]

    #load the test data
    test_X = get_test_data_X()
    test_X = test_X[['WS10']]
    test_Y = get_test_data_Y()

    #1. Linear regression
    #NOTE: fit_intercept=True by default. If the data is centered, set to False
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_X, train_Y)

    predictions = lin_reg_model.predict(test_X)
    score = lin_reg_model.score(test_X, test_Y)
    print(score)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X.index, columns=['POWER']).to_csv("out/part_1/ForecastTemplate1-LR.csv")

    #2. KNN regression
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(train_X, train_Y)

    predictions = knn_model.predict(test_X)
    score = knn_model.score(test_X, test_Y)
    print(score)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X.index, columns=['POWER']).to_csv("out/part_1/ForecastTemplate1-kNN.csv")

    #3. SVR regression
    #TODO: Tune hyperparameters
    svr_model = SVR()
    svr_model.fit(train_X, train_Y)

    predictions = svr_model.predict(test_X)
    score = svr_model.score(test_X, test_Y)
    print(score)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X.index, columns=['POWER']).to_csv("out/part_1/ForecastTemplate1-SVR.csv")

    #4. Neural network regression
    #TODO: Tune hyperparameters
    neural_network = MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=1000, activation='relu')
    neural_network.fit(train_X, train_Y)

    predictions = neural_network.predict(test_X)
    score = neural_network.score(test_X, test_Y)
    print(score)

    #Save the predictions to a .csv file, with the same timestamp as the test data
    pd.DataFrame(predictions, index=test_X.index, columns=['POWER']).to_csv("out/part_1/ForecastTemplate1-NN.csv")

    #5. use a table to compare the value of RMSE error metric among all four machine learning techniques

    """
    6.  for each machine learning technique, please plot a figure for the whole month
    11.2013 to compare the true wind energy measurement and your predicted results. In
    each figure, there are two curves. One curve shows the true wind energy measurement
    and the other curve show the wind power forecasts results
    """
    return None

if __name__ == "__main__":
    part_1()