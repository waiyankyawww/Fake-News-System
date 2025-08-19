# main.py
import torch
import joblib
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


# PROJECT_ROOT = Path(".").resolve()
# DATA_DIR = PROJECT_ROOT / "data"
# csv_path = DATA_DIR / "processed" / "combined_all_with_extra.csv"

# # 1. Load and preprocess data
# X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

# # 2. Extract all features
# feature_funcs = {
#     "BoW": bow_features,
#     "TFIDF": tfidf_features,
#     "Word2Vec": w2v_features,
#     "GloVe": glove_features,
#     # below are too heavy to run on the CPU
#     # "BERT": bert_features,  
#     # "RoBERTa": roberta_features,
#     # "XLNet": xlnet_features
# }

# features = {}
# all_results = {}
# trained_models = {}

# for feat_name, func in feature_funcs.items():
#     print(f"\nExtracting feature: {feat_name}")
#     X_tr, X_te, model_or_emb = func(X_train, X_test)
#     features[feat_name] = (X_tr, X_te, model_or_emb)
#     trained_models[feat_name] = model_or_emb
    
#     print(f"Training models on feature: {feat_name} ...")
#     trained_models_dict, results_dict = train_all_models(X_tr, y_train)

#     print("this is the type of vars", type(trained_models_dict), type(results_dict))
#     print("this is the result dict ", results_dict)
#     print("this is the trained model dict ", trained_models_dict)
 
#     # store them in your main dictionaries
#     all_results[feat_name] = results_dict
#     trained_models[feat_name] = trained_models_dict


#     print(f"\nResults for Feature: {feat_name}")
#     for model_name, metrics in results_dict.items():
#         print(f"{model_name} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")



# List of features
feature_list = ["BoW", "TFIDF", "Word2Vec", "GloVe"] 

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




# 3. User input prediction
while True:
    user_text = input("\nEnter text to check if it is Fake or Real news (or type 'exit' to quit): ")

    if user_text.lower() == "exit":
        print("Exiting program. Goodbye! 👋")
        break

    user_text_clean = clean_text(user_text)
    overall_votes = []
    embedding_features = ["GloVe"]

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

    # Determine overall verdict
    vote_count = Counter(overall_votes)
    final_result = vote_count.most_common(1)[0][0]
    label_map = {0: "real", 1: "fake"}
    final_result_str = label_map[final_result]
    emoji = "✅" if final_result_str == "real" else "❌"
    print(f"\nOverall Result: {emoji} The news is likely {final_result_str}")


# # 3. User input prediction
# while True:
#     user_text = input("\nEnter text to check if it is Fake or Real news (or type 'exit' to quit): ")

#     if user_text.lower() == "exit":
#         print("Exiting program. Goodbye! 👋")
#         break

#     user_text_clean = clean_text(user_text)

#     print("\nPredictions for your input:\n")
#     overall_votes = []

#     # print("This is the features items ", features)

#     for feat_name, data in features.items():
#         print(f"Feature: {feat_name}")
#         # print("this is the data", data)
#         if feat_name in ["BoW", "TFIDF"]:
#             # For BoW/TFIDF: data = (vectorizer, X_train, X_test)
#             X_tr, X_te, vectorizer = data
#             print("this is the X_tr", X_tr)
#             print("this is the X_te", X_te)
#             print("this is the vectorizer", vectorizer)

#             feature_models = trained_models[feat_name]
#             user_vector = vectorizer.transform([user_text_clean])
#         else:
#             # For embeddings: data = (X_train, X_test, model_or_emb)
#             X_tr, X_te, _ = data
#             # Use placeholder zero vector with correct shape
#             user_vector = np.zeros((1, X_tr.shape[1]))
        
#         feature_models = trained_models[feat_name]

#         for model_name, model in feature_models.items():
#             print(f"Predicting with {model_name}...")

#             # Convert sparse to dense if needed
#             if hasattr(user_vector, "toarray"):
#                 user_input = user_vector.toarray()
#             else:
#                 user_input = user_vector

#             if model_name == "SimpleNN":
#                 print(f"going to predict for {model_name} inside if")
#                 # PyTorch model
#                 user_tensor = torch.tensor(user_input, dtype=torch.float32)
#                 model.eval()  # evaluation mode
#                 with torch.no_grad():
#                     outputs = model(user_tensor)
#                     prediction = torch.argmax(outputs, dim=1).item()
#             else:
#                 print(f"going to predict for {model_name} inside else")
#                 # sklearn models
#                 prediction = model.predict(user_input)[0]

#             print(f"{model_name} -> Prediction: {prediction}")
#             overall_votes.append(prediction)

#     print("this is the overall votes", len(overall_votes))
#     print("this is the overall votes", overall_votes)
#     # Determine overall verdict
#     vote_count = Counter(overall_votes)
#     print("This is the vote count ", vote_count)
#     final_result = vote_count.most_common(1)[0][0]
#     # Map numeric prediction to string label
#     label_map = {0: "real", 1: "fake"}
#     final_result_str = label_map[final_result]  # convert 0/1 to "real"/"fake"
#     emoji = "✅" if final_result_str == "real" else "❌"
#     print(f"\nOverall Result: {emoji} The news is likely {final_result_str}")
