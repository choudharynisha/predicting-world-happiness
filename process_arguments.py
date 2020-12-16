"""
Processes the optional command line arguments to get start and end years to
determine which years' happiness data to look at and what seed to use. Uses the
arguments to read the corresponding CSV files.
Author – Nisha Choudhary
Date   – Sunday, December 6, 2020
"""

import argparse
import sys
import load_data

START = 2015
END = 2019

def parse_arguments():
    """Parses the command line arguments provided by the user in order to use
       the correct start and end years, if specified

    Returns:
        int: The first year to look at
        int: The last year to look at
    """
    parser = argparse.ArgumentParser(description = "run linear regression on" +\
        " happiness data")
    parser.add_argument("-s", "--year_start", default = START, type = int,\
        help = "the first year of happiness data to look at (" + str(START) +\
            " or later)", metavar = "")
    parser.add_argument("-e", "--year_end", default = END, type = int,\
        help = "the last year of happiness data to look at (" + str(END) +\
            " or earlier)", metavar = "")
    parser.add_argument("-d", "--seed", default = None, type = int,\
        help = "the seed to run the models at", metavar = "")
    arguments = parser.parse_args()

    if (arguments.year_start < START) or\
        (arguments.year_start > END):
        print("Invalid start year")
        parser.print_help()
        sys.exit()

    if (arguments.year_end < START) or\
        (arguments.year_end > END):
        print("Invalid end year")
        parser.print_help()
        sys.exit()

    if arguments.year_end < arguments.year_start:
        print("Start year cannot be after end year")
        parser.print_help()
        sys.exit()
    
    return arguments.year_start, arguments.year_end, arguments.seed

def read_csv(start, end):
    """Uses the start and end years to determine how much happiness data to
       collect for the model

    Args:
        start (int): The first year of happiness to look at
        end (int): The last year of happiness to look at

    Returns:
        pandas.DataFrame: The DataFrame storing the happiness data of over 150
                          countries over (end - start + 1) years
    """
    data = load_data.load_data(start, end)
    return data

if __name__ == "__main__":
    start, end, seed = parse_arguments()
    print("year_start = {}, year_end = {}, and seed = {}".format(start, end,\
        seed))
