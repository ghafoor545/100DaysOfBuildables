# train_model.py
# Run this ONCE in PyCharm to train and save the model

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

print("Downloading and training model...")

# Load data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# Binary target: good if quality >= 6
df['is_good'] = (df['quality'] >= 6).astype(int)
X = df.drop(['quality', 'is_good'], axis=1)
y = df['is_good']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline with scaling + XGBoost
model = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Train
model.fit(X_train, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")
print("App ready! Now run: streamlit run main.py")