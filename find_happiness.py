"""
Use linear regression in order to measure overall happiness for over 150 countries
Author – Nisha Choudhary
Date   – Sunday, December 6, 2020
"""

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# personal functions
from regression_linear import *
from knr import *
import process_arguments
import load_data

def main():
    start, end, seed = process_arguments.parse_arguments()
    dataframe = process_arguments.read_csv(start, end)
    column_names = ["Country", "Year", "Region", "Happiness Score",
                    "Economy (GDP per Capita)", "Health (Life Expectancy)",
                    "Freedom", "Trust (Government Corruption)", "Generosity"]
    
    labels, data = load_data.split_data(dataframe, column_names)
    
    # use the MinMaxScaler to scale all features between 0 and 1
    scaler = MinMaxScaler()
    features_minmax = pd.DataFrame(data = data)
    features_minmax = scaler.fit_transform(features_minmax)
    data_train, data_test, labels_train, labels_test =\
        load_data.split_train_test(features_minmax, labels, seed)
    
    # LINEAR REGRESSION MODEL
    print("LINEAR REGRESSION MODEL")
    linear_regression =\
        LinearRegressionModel(data_test, labels_test, data_train, labels_train,\
            start, end)
    linear_regression.train()
    linear_regression.predict()
    linear_regression.graph()
    print("R^2 = " + str(linear_regression.r_squared()))
    print("Weights = " + str(linear_regression.weights()))
    
    # KNEIGHBORS REGRESSOR MODEL
    print("\nKNEIGHBORS REGRESSOR MODEL")
    knregressor =\
        KNeighborsRegressorModel(data_test, labels_test, data_train, labels_train,\
            start, end)
    knregressor.train()
    knregressor.predict()
    knregressor.graph()
    print("R^2 = " + str(knregressor.r_squared()))

if __name__ == "__main__":
    main()