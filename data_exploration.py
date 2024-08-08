import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())
