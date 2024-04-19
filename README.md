# OpenAIr
ADA24-Project
Welcome to OpenAIr, a comprehensive analysis of amenities impact on Airbnb success using the [InsideAirbnb dataset](https://insideairbnb.com/get-the-data/).

# Getting Started
For the final project results, once git cloned on your platform, cd into the Final Project/FinalProject_32.ipynb file and make sure the filtered_file.csv is available. 
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
