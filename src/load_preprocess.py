# load_preprocess.py
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    print("Total labels before mapping:\n", df['label'].value_counts())

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Handle both numeric and string labels
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.strip().str.upper()
        df = df[df['label'].isin(['FAKE', 'REAL'])]
        df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    else:
        df['label'] = df['label'].astype(int)

    # Drop any rows with empty text or missing labels
    df = df.dropna(subset=['label', 'text'])
    df = df[df['text'].str.strip() != '']

    print("Total labels after mapping:\n", df['label'].value_counts())
    print(f"Rows remaining: {len(df)}")

    if df.empty:
        raise ValueError("No valid rows found after preprocessing. Check your CSV file.")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = load_and_preprocess("fake_news_dataset.csv")
#     print("Sample preprocessed text:", X_train.iloc[0])
