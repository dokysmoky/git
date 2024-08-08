import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv') 

# Inspect the initial dataset
print("Initial dataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\n")

def convert_to_numeric(val):
    if isinstance(val, str):
        val = val.replace(',', '')  # Remove commas
        if 'k' in val:
            return float(val.replace('k', '').strip()) * 1e3
        elif 'M' in val:
            return float(val.replace('M', '').strip()) * 1e6
        else:
            # Remove any non-numeric text and convert to float
            numeric_part = ''.join(filter(str.isdigit, val))
            return float(numeric_part) if numeric_part else 0
    return val

# Apply the conversion function
df['num_ratings'] = df['num_ratings'].apply(convert_to_numeric)
df['num_reviews'] = df['num_reviews'].apply(convert_to_numeric)
df['num_followers'] = df['num_followers'].apply(convert_to_numeric)

# Handle Missing Values
# Drop rows where the 'rating' or 'genre' are missing, as they might be critical
df.dropna(subset=['rating', 'genre'], inplace=True)

# Fill missing 'num_followers' with 0, as it makes sense if an author has no recorded followers
df['num_followers'] = df['num_followers'].fillna(0)

# Fill missing 'num_reviews' and 'num_ratings' with the median, as these columns likely have outliers
df['num_reviews'] = df['num_reviews'].fillna(df['num_reviews'].median())
df['num_ratings'] = df['num_ratings'].fillna(df['num_ratings'].median())

# Fill missing 'synopsis' with an empty string, as it will be processed later in NLP tasks
df['synopsis'] = df['synopsis'].fillna('')

# Data Consistency and Type Conversions
# Convert 'rating' to a numeric type, ensuring it's within the valid range (0-5)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]

# Ensure 'num_followers', 'num_reviews', and 'num_ratings' are integers (after filling NaNs)
df['num_followers'] = df['num_followers'].astype(int)
df['num_reviews'] = df['num_reviews'].astype(int)
df['num_ratings'] = df['num_ratings'].astype(int)

# Check if any remaining missing values
print("\nPost-cleanup missing values per column:")
print(df.isnull().sum())

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)

# Final output to verify
print("Cleaned dataset info:")
print(df.info())
print("\nSample cleaned data:")
print(df.head())
