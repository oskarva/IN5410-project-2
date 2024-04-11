import pandas as pd
#Any functions common for two or more files may be stored here

def get_train_data() -> pd.DataFrame: 
    train_data = pd.read_csv("TrainData.csv")
    train_data.set_index('TIMESTAMP', inplace=True)
    return train_data

def get_test_data_X() -> pd.DataFrame:
    test_data_X = pd.read_csv("WeatherForecastInput.csv")
    test_data_X.set_index('TIMESTAMP', inplace=True)
    return test_data_X

def get_test_data_Y() -> pd.DataFrame:
    test_data_Y = pd.read_csv("Solution.csv")
    test_data_Y.set_index('TIMESTAMP', inplace=True)
    return test_data_Y