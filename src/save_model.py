import joblib
import torch
from load_preprocess import load_and_preprocess, clean_text
from feature_extraction import (
    bow_features, tfidf_features, w2v_features, glove_features,
    bert_features, roberta_features, xlnet_features
)
from train_models import train_all_models
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter


PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
csv_path = DATA_DIR / "processed" / "combined_all_with_extra.csv"

# 1. Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

# 2. Extract all features
feature_funcs = {
    "BoW": bow_features,
    "TFIDF": tfidf_features,
    # "Word2Vec": w2v_features,
    # "GloVe": glove_features,
    # below are too heavy to run on the CPU
    # "BERT": bert_features,  
    # "RoBERTa": roberta_features,
    # "XLNet": xlnet_features
}

features = {}
all_results = {}
trained_models = {}


for feat_name, func in feature_funcs.items():
    print(f"\nExtracting feature: {feat_name}")
    X_tr, X_te, model_or_emb = func(X_train, X_test)
    

    # Save the feature transformer (e.g., BoW, TFIDF)
    joblib.dump(model_or_emb, f"{feat_name}_transformer.pkl")
    
    print(f"Training models on feature: {feat_name} ...")
    trained_models_dict, results_dict = train_all_models(X_tr, y_train)
    
    # Save the trained models
    joblib.dump(trained_models_dict, f"{feat_name}_models.pkl")
    
    print(f"\nResults for Feature: {feat_name}")
    for model_name, metrics in results_dict.items():
        print(f"{model_name} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    # Optional: save combined results for this feature
    joblib.dump(results_dict, f"{feat_name}_results.pkl")

print("all done")



