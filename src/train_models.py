# train_models.py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Simple Feedforward DL model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

def train_all_models(X, y):
    

    # Split dataset
    print("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if sparse or dense
    print("Checking Spars or dense")
    is_sparse = hasattr(X_train, "toarray")
    X_train_dense = X_train.toarray() if is_sparse else X_train
    X_test_dense = X_test.toarray() if is_sparse else X_test

    # Optional: reduce dimensions for dense embeddings
    if not is_sparse and X_train_dense.shape[1] > 200:
        print("⚡ Applying PCA to reduce dimensionality...")
        pca = PCA(n_components=100, random_state=42)
        X_train_dense = pca.fit_transform(X_train_dense)
        X_test_dense = pca.transform(X_test_dense)

    # Choose faster SVM
    svm_model = LinearSVC(max_iter=2000) if is_sparse else SVC(kernel='linear')

    
    # ML Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "SVM": svm_model,
        "RandomForest": RandomForestClassifier(
            n_estimators=50,       # fewer trees
            max_depth=15,          # limit depth
            max_features='sqrt',   # consider subset of features
            n_jobs=-1,             # parallelize across CPU cores
            random_state=42
        ),
        "NaiveBayes": GaussianNB() if not is_sparse else MultinomialNB()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print("training ", {model})
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        trained_models[name] = model  # save trained model

    # Deep Learning Model
    print("going to train DL models")
    X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    dl_model = SimpleNN(X_train_dense.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.001)


    print("going to start train_loader loop")
    for epoch in range(5):  # small epoch for example
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = dl_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate DL model
    print("evaluate DL models")
    with torch.no_grad():
        outputs = dl_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        results["SimpleNN"] = {
            "accuracy": accuracy_score(y_test_tensor, predicted),
            "f1_score": f1_score(y_test_tensor, predicted, average='weighted')
        }
        trained_models["SimpleNN"] = dl_model  # save DL model

    print ("This is the trained models", trained_models)
    print ("This is the results", results)
    return trained_models, results

