# Predicting World Happiness

## Description
For my final project for Machine Learning, I looked at 2015 through 2019 happiness data from [the World Happiness Report](https://worldhappiness.report). To predict a country's happiness score from a particular year, I implemented Linear Regression and K Neighbors Regressor from scikit-learn.

## Running the Program
The main driver program for this project is `find_happiness.py`.

### Requirements
The language requirement is Python 3. To download Python 3, please visit [the Download Python page](https://www.python.org/downloads).

To run `find_happiness.py`, you need the following Python packages –<br />
– `matplotlib`<br />
– `numpy`<br />
– `pandas`<br />
– `scikit-learn`

To download these packages with pip, you can use the following commands in your Terminal / Command Prompt window –<br />
`pip3 install matplotlib`<br />
`pip3 install numpy`<br />
`pip3 install pandas`<br />
`pip3 install scikit-learn`

If you do not have pip installed, please navigate to [the pip documentation's Installation page](https://pip.pypa.io/en/stable/installing).

### Command Line Arguments
All command line arguments are optional. Entering `python3 find_happiness.py` will run the program using data from 2015 through 2019 and split the data with a random seed.

#### Optional Arguments
`-s` (`year_start`) – The first year's data to look at (2015 through 2019, inclusive) \[DEFAULT = 2015\]<br />
`-e` (`year_end`) – The last year's data to look at (2015 through 2019, inclusive) \[DEFAULT = 2019\]<sup>\*</sup><br />
`-d` (`seed`) – The specified seed to split the train and test data on \[DEFAULT = `None`\]<sup>\*\*</sup>

Examples of commands to run with the optional commands –<br />
`python3 find_happiness.py -s 2016 -e 2018 -d 18`<br />
*Runs the program only using data from 2016 through 2018 with a seed of 18*

`python3 find_happiness.py -e 2017`<br />
*Runs the program only using data from 2015 through 2017 with a random seed*

`python3 find_happiness.py -s 2019 -d 42`<br />
*Runs the program only using data from 2019 with a seed of 42*

<sup>\*</sup> The program will not run if the specified `year_end` is before the specified `year_start`.<br />
<sup>\*\*</sup> If the seed is `None`, a random seed will be used to split the data into training and testing.

### Output
The predicted happiness scores versus the actual happiness scores will be graphed and displayed in a new window that opens up upon running the program (with the first graph's window displayed after a short delay). The K Neighbors Regressor's graph will open in a new window after the graph for Linear Regression is closed and another short delay. In all of the scatterplots, each point represents a country's prediction vs actual label from a country between `year_start` and `year_end`, inclusive. When looking back at the Terminal window or Command Prompt after closing the graphs, it shows the R<sup>2</sup> coefficient for the Linear Regression and the K Neighbors Regressor models and the weights for the Linear Regression model.

To find out which country a point represents, hover over it when the graph's window is shown.

#### Regions
The regions on the graph are numbered as follows –<br />
0 = Australia and New Zealand<br />
1 = Central and Eastern Europe<br />
2 = Eastern Asia<br />
3 = Latin America and Carribean<br />
4 = Middle East and Northern Africa<br />
5 = North America<br />
6 = Southeastern Asia<br />
7 = Southern Asia<br />
8 = Sub-Saharan Africa<br />
9 = Western Europe

## Dataset
I found the 2015 through 2020<sup>\*</sup> data from the World Happiness Report on [Kaggle](https://www.kaggle.com/mathurinache/world-happiness-report).

The columns<sup>\*\*</sup> I looked at from the dataset were Country, Region<sup>\*\*\*</sup>, Happiness Score, Economy (GDP per Capita), Health (Life Expectancy), Freedom, Trust (Government Corruption), and Generosity.

*Health (Life Expectancy) –* based on the data extracted from the World Health Organization's Global Health Observatory data repositoory.<sup>\*\*\*\*</sup>

*Freedom –* national average of responses to “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”

*Trust (Government Corruption)<sup>\*\*\*\*\*</sup> –* The questions asked were, “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?” The overall perception is just the average of the two 0-or-1 responses.

*Generosity –* national average of response to the Gallup World Poll question “Have you donated money to a charity in the past month?”

The dataset is stored in a Pandas DataFrame. Certain years did not have region, so the regions used in the 2015 dataset are mapped to every country in all of the other datasets. An additional column was added to the DataFrame in order to make note of the year that this datapoint is from in order to concatenate each year's DataFrame into one larger DataFrame while retaining the information about the year it came from.

<sup>\*</sup> Because of the differences and inconsistencies in the 2020 data, I used the 2015 through 2019 data from this dataset.<br />
<sup>\*\*</sup> The column names used are from the 2015 dataset and combined with equivalent columns from other datasets.<br />
<sup>\*\*\*</sup> Regions used are from the 2015 dataset and are mapped to all countries in the other datasets to ensure consistency and add regions to datasets that didn't previously have them<br />
<sup>\*\*\*\*</sup> The data at the source are available for the years 2000, 2005, 2010, 2015 and 2016. To match the report’s sample period, interpolation and extrapolation are used.<br />
<sup>\*\*\*\*\*</sup> Labeled as Corruption Perception in some datasets

## Observations
The seeds that I compared the results of were 10, 18, 26, 34, and 42.

### Common observations among all seeds
1. The K Neighbors Regressor model was more correlated than the Linear Regression model, no matter what the seed was for shuffling, even though linear regression is known to work well with continuous data and K Nearest Neighbors is not known to be a very effective algorithm.
2. While there were a few outliers that predicted a lower happiness score than it should have, most outliers would predict a higher happiness score, regardless of whether the model was K Neighbors Regressor or Linear Regression.
3. As the seed increased, the correlation coefficient R<sup>2</sup> for both models decreased.
4. The models performed about the same, regardless of whether or not the feature values were scaled using the MinMaxScaler.
5. Both models had a generally high correlation coefficient R<sup>2</sup> (> 0.7), and most points on the scatterplots were somewhere around the ideal line of `y = x`.

__*Seed of 10*__<br />
*Linear Regression*<br />
– R<sup>2</sup> ≈ 0.75506<br />
– Weights<br />
&emsp;→ Economy (GDP per Capita) ≈ 2.77764<br />
&emsp;→ Health (Life Expectancy) ≈ 1.33523<br />
&emsp;→ Freedom ≈ 1.45193<br />
&emsp;→ Trust (Government Corruption) ≈ 0.279344<br />
&emsp;→ Generosity ≈ 0.29783

*K Neighbors Regressor*<br />
– R<sup>2</sup> ≈ 0.81473

__*Seed of 18*__<br />
*Linear Regression*
– R<sup>2</sup> ≈ 0.79730<br />
– Weights<br />
&emsp;→ Economy (GDP per Capita) ≈ 2.84379<br />
&emsp;→ Health (Life Expectancy) ≈ 1.27833<br />
&emsp;→ Freedom ≈ 1.25342<br />
&emsp;→ Trust (Government Corruption) ≈ 0.25597<br />
&emsp;→ Generosity ≈ 0.45937

*K Neighbors Regressor*<br />
– R<sup>2</sup> ≈ 0.85948

__*Seed of 26*__<br />
*Linear Regression*
– R<sup>2</sup> ≈ 0.75742<br />
– Weights<br />
&emsp;→ Economy (GDP per Capita) ≈ 2.84740<br />
&emsp;→ Health (Life Expectancy) ≈ 1.29336<br />
&emsp;→ Freedom ≈ 1.36589<br />
&emsp;→ Trust (Government Corruption) ≈ 0.18373<br />
&emsp;→ Generosity ≈ 0.37860

*K Neighbors Regressor*<br />
– R<sup>2</sup> ≈ 0.81424

__*Seed of 34*__<br />
*Linear Regression*
– R<sup>2</sup> ≈ 0.71898<br />
– Weights<br />
&emsp;→ Economy (GDP per Capita) ≈ 2.847401<br />
&emsp;→ Health (Life Expectancy) ≈ 1.26220<br />
&emsp;→ Freedom ≈ 1.42733<br />
&emsp;→ Trust (Government Corruption) ≈ 0.35726<br />
&emsp;→ Generosity ≈ 0.32781

*K Neighbors Regressor*<br />
– R<sup>2</sup> 0.78445

__*Seed of 42*__<br />
*Linear Regression*<br />
– R<sup>2</sup> ≈ 0.71544<br />
– Weights<br />
&emsp;→ Economy (GDP per Capita) ≈ 2.631697<br />
&emsp;→ Health (Life Expectancy) ≈ 1.38927<br />
&emsp;→ Freedom ≈ 1.38302<br />
&emsp;→ Trust (Government Corruption) ≈ 0.46644<br />
&emsp;→ Generosity ≈ 0.29325

*K Neighbors Regressor*
– R<sup>2</sup> ≈ 0.78414

### Other Observations
1. Economy was more correlated to how happy a country's average citizen perceived themselves to be than personal freedom.
2. When the seed is lower, generosity is more correlated with a country's average citizen's perception of personal happiness than corruption. When the seed is higher, this is reversed.
3. Western European countries tend to have the highest happiness scores and predictions while Sub-Saharan African countries tend to have the lowest happiness scores.
4. Even though the region was part of the label and not one of the features, the models were still able to identify patterns that different countries in each region had, and most countries in each region are in the same part of the graph.

### Other Interpretations
1. Before having each point on the scatterplot colored based on which region the country was in, it seemed like the models were working well, and the data I found on Kaggle was good data. When looking at the graphs after coloring each point by region, it seems like the data may look at each country's data to determine and assess its overall happiness score only based on the happiness definition of Western Europe.
2.  Countries with darker skinned / black populations tend to be on the lower end of the ranking. It makes me curious about how to change this dataset to include multiple definitions of happiness and taking in the different ways that different cultures and different individuals perceive think about what it means to be happy and other factors that can be measured instead.
3. Going off of the last observation about how a country's region was not a feature used to predict the happiness score of each country, it seems like a lot of this computation was based off a definition that a certain group of people, and by this definition, since different countries in the same region do have similarities, it seems to make sense, in a way, that countries in the same region may be in the same part of the scatterplots.

## Goals for the Future
– Update the dataset and predict 2021 data (and later)<br />
– Learn more about how to collect data<br />
– Put together different data sources<br />
&emsp;→ Consider other potential factors of happiness, including but not limited to<br />
&emsp;&emsp;⇒ Volunteer work<br />
&emsp;&emsp;⇒ Scale of self-appreciation<br />
&emsp;&emsp;&emsp;⇛ 1 = no self-appreciation<br />
&emsp;&emsp;&emsp;⇛ 10 = lots of self-appreciation<br />
&emsp;&emsp;⇒ How much natural light people are exposed to<br />
&emsp;&emsp;⇒ Air quality and pollution – scale<br />
&emsp;&emsp;&emsp;⇛ 1 = no pollution<br />
&emsp;&emsp;&emsp;⇛ 10 = high pollution<br />
&emsp;&emsp;⇒ Personal assessment of work-life balance<br />
&emsp;&emsp;⇒ How much experiences versus material goods are valued<br />
&emsp;→ Compare similar measures from different data sources<br />
– Learn more about positive psychology to help determine what leads to better perceptions of personal happiness<br />
– Make a web app to make the data and results more accessible<br />
– Combine with [the Happy Journal web app project](https://github.com/choudharynisha/CS380-Happy-Journal)