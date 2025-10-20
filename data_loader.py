import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Define column names
    DEFAULT_COLS = ["variance", "skewness", "curtosis", "entropy", "class"]
    TARGET = "class"

    # Load from UCI URL
    df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
        names=DEFAULT_COLS
    )

    # Check columns
    missing_cols = [c for c in DEFAULT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. Expected: {DEFAULT_COLS}")

    # Handle missing values (median for numeric columns)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Split features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Train-test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return df, X_train, X_test, y_train, y_test