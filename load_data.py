"""
Loads in, processes, and stores the happiness data
Author = Nisha Choudhary
Date   = Friday, December 4, 2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataframe = load_data(2015, 2019)
    dataframe.describe()

    for column in dataframe.columns:
        # graph all columns with numberic / continuous variables
        if column != "Region":
            plt.title(column)
            plt.hist(dataframe[column], bins = 20)
            plt.show()
    
    print(dataframe["Region"].value_counts())

def load_data(starting_year, ending_year):
    """Process the loading in of the data to store in a Pandas DataFrame

    Returns:
        pandas.DataFrame: The DataFrame storing the happiness data of over 150
                          countries over several years
    """
    regions, columns, rename = get_standard_information()
    dataframe = read_files(starting_year, ending_year, rename, regions, columns)
    
    return dataframe

def get_standard_information():
    """Sets the desired column names in order to process and standardize the data

    Returns:
        list: The regions of each country in the dataset
        list: The column names from the datasets to be kept
        list: The names to be used for non-standard data
    """
    dataframe_ref = pd.read_csv("data/2015.csv")

    # getting the region in order to add it to other datasets and correcting the
    # errors
    regions = dict(zip(dataframe_ref["Country"], dataframe_ref["Region"]))
    regions["Namibia"] = "Sub-Saharan Africa"
    regions["Northern Cyprus"] = regions["Cyprus"]
    regions["South Sudan"] = "Sub-Saharan Africa"
    regions["Somalia"] = "Sub-Saharan Africa"
    regions["Trinidad & Tobago"] = "Latin America and Caribbean"
    regions["Gambia"] = "Sub-Saharan Africa"
    regions["North Macedonia"] = regions["Albania"]
    regions["Belize"] = "Latin America and Caribbean"
    regions["Taiwan Province of China"] = "Eastern Asia"
    regions["Hong Kong S.A.R., China"] = "Eastern Asia"

    columns = ["Country", "Year", "Region", "Happiness Score",
               "Economy (GDP per Capita)", "Health (Life Expectancy)",
               "Freedom", "Trust (Government Corruption)", "Generosity"]

    rename = {"Happiness.Score": "Happiness Score",
              "Economy..GDP.per.Capita.": "Economy (GDP per Capita)",
              "Health..Life.Expectancy.": "Health (Life Expectancy)",
              "Trust..Government.Corruption.": "Trust (Government Corruption)",
              "Score": "Happiness Score",
              "GDP per capita": "Economy (GDP per Capita)",
              "Social support": "Family",
              "Healthy life expectancy": "Health (Life Expectancy)",
              "Freedom to make life choices": "Freedom",
              "Perceptions of corruption": "Trust (Government Corruption)",
              "Regional indicator": "Region"
             }

    return regions, columns, rename

def read_files(starting_year, ending_year, renamed, regions, columns):
    """Reads in the data from all of the CSV files

    Args:
        starting_year (int): The first year that the data is coming from
        ending_year (int): The last year that the data is coming from
        renamed (list): The names to be used for non-standard data
        regions (list): The regions of each country in the dataset
        columns (list): The column names from the datasets to be kept

    Returns:
        pandas.dataframe: All data from each of the years from all of the
                          columns wanted
    """
    dataframes = []

    for year in range(starting_year, ending_year + 1):
        file_name = "data/" + str(year) + ".csv"
        dataframe = pd.read_csv(file_name).rename(columns = renamed)
        
        # making sure that country and region is in every year's dataset
        if "Country or region" in dataframe.columns:
            dataframe.rename(columns = {"Country or region": "Country"},\
                inplace = True)
        
        if "Region" not in dataframe.columns:
            # adding each country's region to all post-2015 data
            dataframe["Region"] = dataframe["Country"].map(regions)
        
        # making note of the year in the data for reference in aggregated data
        dataframe["Year"] = year

        # temporary fix for filling in the missing Trust (Government Corruption)
        # value for the United Arab Emirates in 2018 with a 0 and assumes that
        # any future dataset added will be a feature value (not a region,
        # country name, or happiness score)
        dataframe = dataframe.fillna(0)

        if np.sum(dataframe.isna()).sum() > 0:
            # last check before filling in the dataframe's missing values with
            # 0 to the list to check where future datasets added may have any
            # missing values
            print(np.sum(dataframe.isna()))

        dataframes.append(dataframe[columns])

    # putting each year's data together
    dataframe = pd.concat(dataframes)
    return dataframe

if __name__ == "__main__":
    main()
