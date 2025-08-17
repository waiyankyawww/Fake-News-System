# feature_extraction.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel
import torch
from pathlib import Path

# BoW
def bow_features(train_texts, test_texts):
    print("inside bow features function")
    vectorizer = CountVectorizer(max_features=5000)
    X_train_bow = vectorizer.fit_transform(train_texts)
    X_test_bow = vectorizer.transform(test_texts)
    return X_train_bow, X_test_bow, vectorizer

# TF-IDF
def tfidf_features(train_texts, test_texts):
    print("inside tfidf features function")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Word2Vec
def w2v_features(train_texts, test_texts):
    print("inside w2v features function")
    tokenized_train = [text.split() for text in train_texts]
    tokenized_test = [text.split() for text in test_texts]
    
    # Train Word2Vec model on training data only
    model = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=1, workers=4)
    
    def vectorize(tokenized_docs):
        return np.array([
            np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
            for words in tokenized_docs
        ])
    
    X_train_w2v = vectorize(tokenized_train)
    X_test_w2v = vectorize(tokenized_test)
    
    return X_train_w2v, X_test_w2v, model

# # GloVe
# def glove_features(texts, glove_path="data/raw/glove/glove.6B.100d.txt"):
#     print("inside glove features function")
#     embeddings = {}
#     with open(glove_path, "r", encoding="utf-8") as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], dtype="float32")
#             embeddings[word] = vector

#     tokenized = [text.split() for text in texts]
#     X = np.array([np.mean([embeddings[word] for word in words if word in embeddings] or [np.zeros(100)], axis=0) for words in tokenized])
#     return X, embeddings

# GloVe
def glove_features(train_texts, test_texts):
    print("inside glove features function")
    glove_path = Path(__file__).parent.parent / "data/raw/glove/glove.6B.100d.txt"
    print("Thsi si the glove path", glove_path)
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector

    def vectorize(tokenized_docs):
        return np.array([
            np.mean([embeddings[word] for word in words if word in embeddings] or [np.zeros(100)], axis=0)
            for words in tokenized_docs
        ])
    
    tokenized_train = [text.split() for text in train_texts]
    tokenized_test = [text.split() for text in test_texts]
    
    X_train_glove = vectorize(tokenized_train)
    X_test_glove = vectorize(tokenized_test)
    
    return X_train_glove, X_test_glove, embeddings

# BERT
def bert_features(texts):
    print("inside bow features function")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    features = []
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = model(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            features.append(cls_embedding.squeeze())
    return np.array(features), model

# BERT
def bert_features(train_texts, test_texts):
    print("inside bert features function")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_features(texts):
        tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    X_train = get_features(train_texts)
    X_test = get_features(test_texts)

    return X_train, X_test, (model, tokenizer)


# RoBERTa
def roberta_features(train_texts, test_texts):
    print("inside roberta features function")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    def get_features(texts):
        tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    X_train = get_features(train_texts)
    X_test = get_features(test_texts)

    return X_train, X_test, (model, tokenizer)


# XLNet
def xlnet_features(train_texts, test_texts):
    print("inside xlnet features function")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')

    def get_features(texts):
        tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    X_train = get_features(train_texts)
    X_test = get_features(test_texts)

    return X_train, X_test, (model, tokenizer)

# # RoBERTa
# def roberta_features(texts):
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     model = RobertaModel.from_pretrained('roberta-base')
#     features = []
#     with torch.no_grad():
#         for text in texts:
#             tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#             outputs = model(**tokens)
#             cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
#             features.append(cls_embedding.squeeze())
#     return np.array(features), model

# # XLNet
# def xlnet_features(texts):
#     tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#     model = XLNetModel.from_pretrained('xlnet-base-cased')
#     features = []
#     with torch.no_grad():
#         for text in texts:
#             tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#             outputs = model(**tokens)
#             cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
#             features.append(cls_embedding.squeeze())
#     return np.array(features), model
