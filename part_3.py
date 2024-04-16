from helper_functions import *

def part_3():
    #load the training data
    #train_data = get_train_data()

    #Only need windspeed data (10 meters above ground level)
    train_data = pd.read_csv("TrainData.csv")
    train_X = train_data[['TIMESTAMP']]
    train_Y = train_data[['POWER']]

    #load the test data
    test_X = pd.read_csv("WeatherForecastInput.csv")
    test_X = test_X[['TIMESTAMP']]
    test_Y = get_test_data_Y()

    #1. Linear regression

if __name__ == "__main__":
    part_3()