import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "MAE"],
        "Train": [train_accuracy, train_f1, train_mae],
        "Test": [test_accuracy, test_f1, test_mae]
    }).round(4)

    return metrics_df, y_test_pred