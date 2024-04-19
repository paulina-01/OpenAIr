# OpenAIr
ADA24-Project
Welcome to OpenAIr, a comprehensive analysis of amenities impact on Airbnb success using the [InsideAirbnb dataset](https://insideairbnb.com/get-the-data/).

Welcome to the OpenAIr project! This repository examines the influence of amenities on the success of Airbnb listings in Italy, specifically focusing on the cities of Florence, Naples, and Venice. These cities were chosen due to their similar sizes, approximate data point parity, and comprehensive geographic coverage. Utilizing the [InsideAirbnb dataset](https://insideairbnb.com/get-the-data/) we analyzed over 30,000 raw data points from these cities.

## Project Overview

### Data Collection and Preparation
The Inside Airbnb dataset provided extensive information on Airbnb listings. We conducted thorough data exploration and multicollinearity analysis to ensure a viable correlation. Attributes were meticulously analyzed for relevance, and feature engineering was executed to transform or create new features, such as distance from the city center.

### Analysis Techniques
- **Fuzzy Word Matching:** Used to extract and one-hot encode unique amenities.
- **XGBoost Regression Models:** Employed to predict the listing price and average star review, and to conduct feature importance analysis. 

# Getting Started
For the final project results, once git cloned on your platform, ```cd Final Project/FinalProject_32.ipynb ``` file and make sure the filtered_file.csv is available. 
Run the first few lines of code to make sure the csv file is connected to the notebook.

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
%pip install python-Levenshtein
%pip install fuzzy-wuzzy
```
The complete the imports necessary to run the file: 
```python
import pandas as pd
import os
from tabulate import tabulate
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, norm
from scipy import stats
import plotly.express as px
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
import sklearn.model_selection as ms
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
```
Finally, run the **Data Preprocessing** block to complete setup. After this, the **Modelling** block can be run to go through the final analysis, as well as edit the model being used. 

# Data Exploration

To review the Data Exploration milestone, ```cd Milestone2/351_M2_Project.ipynb ``` and run all lines to view initial analysis done on the InsideAirbnb Dataset. 
