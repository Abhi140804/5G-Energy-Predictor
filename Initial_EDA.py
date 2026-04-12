# Import Libraries
import pandas as pd
import warnings

# Deal with Warnings
warnings.filterwarnings("ignore")

# Get Dataset
data = pd.read_csv("5G_energy_consumption_dataset.csv")
dataset = pd.read_csv("5G_energy_consumption_dataset.csv")

# Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_more_desc_statistic = data.describe(include = "all")
data_mode = data.mode()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr(numeric_only=True) 
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 10)