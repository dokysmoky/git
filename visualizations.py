import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Histogram for each feature
df.hist(bins=30, figsize=(12, 10), layout=(3, 3))
plt.suptitle('Histograms of Features')
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12, 10))
for i, feature in enumerate(['num_ratings', 'num_reviews', 'num_followers']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=df, y=feature)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, vars=['num_ratings', 'num_reviews', 'num_followers'])
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()
