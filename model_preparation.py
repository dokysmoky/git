#This script will handle the loading of the cleaned dataset, preparing it for modeling, including feature scaling, and splitting the data into training and testing sets.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Inspect the dataset
print("Dataset info:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Define features and target variable
features = ['num_ratings', 'num_reviews', 'num_followers']
target = 'rating'

X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print shapes of the resulting datasets
print("\nTraining set shape:", X_train_scaled.shape)
print("Testing set shape:", X_test_scaled.shape)
