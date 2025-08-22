import streamlit as st
import joblib
import torch
import numpy as np
from collections import Counter
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk
from pathlib import Path


PROJECT_ROOT = Path(".").resolve()
MODEL_DIR = PROJECT_ROOT / "src" / "models"


# List of features
feature_list = ["BoW", "TFIDF", "Word2Vec", "GloVe"] 
embedding_features = ["GloVe"]

# Load all transformers and models
transformers = {}
models_dict = {}

for feat_name in feature_list:
    try:
        transformers[feat_name] = joblib.load(f"models/{feat_name}_transformer.pkl")
        models_dict[feat_name] = joblib.load(f"models/{feat_name}_models.pkl")
        print(f"{feat_name} loaded")
    except FileNotFoundError:
        print(f"Warning: Saved files for {feat_name} not found.")
        transformers[feat_name] = None
        models_dict[feat_name] = None

st.title("📰 Fake News Detector")
st.write("Enter a news headline or text below to check if it is **real or fake**.")


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

# User input
user_text = st.text_area("Enter text:", "")

if st.button("Check News"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        user_text_clean = clean_text(user_text)
        overall_votes = []

        for feat_name in feature_list:
            print(f"\nFeature: {feat_name}")
            transformer = transformers[feat_name]
            feature_models = models_dict[feat_name]

            if transformer is None or feature_models is None:
                print(f"Skipping {feat_name} as it was not loaded properly.")
                continue

            if feat_name in ["BoW", "TFIDF"]:
            # Sparse vectorizer
                user_vector = transformer.transform([user_text_clean])
            elif feat_name == "Word2Vec":
                words = user_text_clean.split()  # simple tokenization
                vec_list = []
                for w in words:
                    if w in transformer.wv:  # check in word vectors
                        vec_list.append(transformer.wv[w])
                if vec_list:
                    user_vector = np.mean(vec_list, axis=0).reshape(1, -1)
                else:
                    user_vector = np.zeros((1, transformer.vector_size))
            elif feat_name in embedding_features:
                # Convert text to embedding vector
                words = user_text_clean.split()  # simple tokenization
                vec_list = []
                for w in words:
                    if w in transformer:  # transformer here is the loaded embedding dict/model
                        vec_list.append(transformer[w])
                if vec_list:
                    user_vector = np.mean(vec_list, axis=0).reshape(1, -1)  # average word vectors
                else:
                    # fallback if no word is in vocabulary
                    user_vector = np.zeros((1, transformer.vector_size))  # adjust vector_size accordingly
            else:
                # Placeholder fallback
                user_vector = np.zeros((1, transformer.shape[1]))

            # Convert sparse to dense if needed
            if hasattr(user_vector, "toarray"):
                user_input = user_vector.toarray()
            else:
                user_input = user_vector

            for model_name, model in feature_models.items():
                print(f"Predicting with {model_name}...")

                if model_name == "SimpleNN":
                    user_tensor = torch.tensor(user_input, dtype=torch.float32)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(user_tensor)
                        prediction = torch.argmax(outputs, dim=1).item()
                else:
                    prediction = model.predict(user_input)[0]

                print(f"{model_name} -> Prediction: {prediction}")
                overall_votes.append(prediction)

        # Final verdict (majority voting)
        vote_count = Counter(overall_votes)
        final_result = vote_count.most_common(1)[0][0]
        label_map = {0: "Real ✅", 1: "Fake ❌"}

        st.subheader(f"### 🏆 Final Prediction is: {label_map[final_result]}")
