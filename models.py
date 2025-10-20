from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_model(model_name):
    if model_name == "Logistic Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=400, random_state=42))
        ])
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=5, min_samples_split=2, criterion="gini", random_state=42
        )
    elif model_name == "KNN":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model