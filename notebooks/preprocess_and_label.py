import pandas as pd
import re

# Load CSV
df = pd.read_csv('data/news_headlines_raw.csv')

# Clean titles
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['clean_title'] = df['title'].apply(clean_text)

# Manual source - bias label map
bias_map = {
    'bbc-news': 'center',
    'cnn': 'left',
    'fox-news': 'right',
    'the-times-of-india': 'center',
    'the-hindu': 'left',
}

# Assign bias label
df['bias'] = df['source'].map(bias_map)

# Drop rows with no label
df = df.dropna(subset=['bias'])

# Save cleaned data
df.to_csv('data/clean_labeled_news.csv', index=False)
print(" Cleaned + labeled data saved to data/clean_labeled_news.csv")
print(df[['clean_title', 'bias']].head())