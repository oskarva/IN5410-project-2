from sklearn.linear_model import LinearRegression

#Training data = TrainData.csv, ONLY 10m above ground level
#relationship between generation and wind speed
#Test data = WeatherForecastInput.csv (test inputs) and Solution.csv (gold standard)
#Generates: four output .csv files 
def part_1():
    #1. Linear regression
    #NOTE: fit_intercept=True by default. If the data is centered, set to False
    lin_reg_model = LinearRegression()
    #lin_reg_model.fit(, y)

    #2. KNN regression

    #3. SVR regression

    #4. Neural network regression

    #5. use a table to compare the value of RMSE error metric among all four machine learning techniques

    """
    6.  for each machine learning technique, please plot a figure for the whole month
    11.2013 to compare the true wind energy measurement and your predicted results. In
    each figure, there are two curves. One curve shows the true wind energy measurement
    and the other curve show the wind power forecasts results
    """
    return None

