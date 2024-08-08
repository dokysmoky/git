import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Function to generate and save word cloud
def generate_wordcloud(text, genre):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Genre: {genre}')
    plt.show()

# Group by genre and generate word clouds
for genre, group in df.groupby('genre'):
    text = ' '.join(group['synopsis'].dropna())
    generate_wordcloud(text, genre)
