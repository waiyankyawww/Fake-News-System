# main.py
from load_preprocess import load_and_preprocess, clean_text
from feature_extraction import (
    bow_features, tfidf_features, w2v_features, glove_features,
    bert_features, roberta_features, xlnet_features
)
from train_models import train_all_models
import numpy as np

# 1. Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess("/Users/waiyankyaw/Desktop/hallam/Dissertation/fake-news-system/data/processed/combined_all.csv")

# 2. Extract all features
feature_funcs = {
    # "BoW": bow_features,
    # "TFIDF": tfidf_features,
    # "Word2Vec": w2v_features,
    # "GloVe": glove_features,
    "BERT": bert_features,
    "RoBERTa": roberta_features,
    "XLNet": xlnet_features
}

features = {}
all_results = {}
trained_models = {}

for feat_name, func in feature_funcs.items():
    print(f"\nExtracting feature: {feat_name}")
    X_tr, X_te, model_or_emb = func(X_train, X_test)
    features[feat_name] = (X_tr, X_te)
    trained_models[feat_name] = model_or_emb
    
    print(f"Training models on feature: {feat_name} ...")
    # results = train_all_models(X_tr, y_train)  # returns dict of model results
    # print("This is the type of results", type(results))
    # print(results)

    # all_results[feat_name] = results["All_Models"]
    # trained_models[feat_name] = results["Trained_Models"]  # Save trained models for user input

    trained_models_dict, results_dict = train_all_models(X_tr, y_train)


    print("this is the type of vars", type(trained_models_dict), type(results_dict))
    print("this is the result dict ", results_dict)
    print("this is the trained model dict ", trained_models_dict)
 
    # store them in your main dictionaries
    all_results[feat_name] = results_dict
    trained_models[feat_name] = trained_models_dict



    print(f"\nResults for Feature: {feat_name}")
    for model_name, metrics in results_dict.items():
        print(f"{model_name} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

# 3. User input prediction
user_text = input("\nEnter text to check if it is Fake or Real news: ")
user_text_clean = clean_text(user_text)

print("\nPredictions for your input:\n")
overall_votes = []

for feat_name, (X_tr, X_te) in features.items():
    print(f"Feature: {feat_name}")
    feature_models = trained_models[feat_name]

    for model_name, model in feature_models.items():
        # Transform user text to feature vector
        if feat_name in ["BoW", "TFIDF"]:
            vectorizer = X_tr  # Assumes X_tr returned as vectorizer for these
            user_vector = vectorizer.transform([user_text_clean])
        else:
            # TODO: Implement embedding transformations for Word2Vec/BERT/GloVe etc.
            user_vector = np.zeros((1, model.coef_.shape[1]))  # Placeholder
        prediction = model.predict(user_vector)[0]
        print(f"{model_name} -> Prediction: {prediction}")
        overall_votes.append(prediction)

# Determine overall verdict
from collections import Counter
vote_count = Counter(overall_votes)
final_result = vote_count.most_common(1)[0][0]
emoji = "✅" if final_result.lower() == "real" else "❌"
print(f"\nOverall Result: {emoji} The news is likely {final_result}")
