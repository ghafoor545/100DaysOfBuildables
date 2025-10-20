# main.py
# Run this to launch the app in browser

import streamlit as st
import pandas as pd
import pickle


# Load model (from same folder)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model()

# App UI
st.set_page_config(page_title="Wine Quality", layout="centered")
st.title("Wine Quality Predictor")
st.markdown("**Enter values → Instant prediction**")

# Input fields
cols = st.columns(3)
features = {}
names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

for i, name in enumerate(names):
    with cols[i % 3]:
        default = 7.0 if 'acidity' in name else 30.0 if 'sulfur' in name else 0.99 if name == 'density' else 0.5
        val = st.number_input(name.replace('_', ' ').title(), value=float(default), step=0.1)
        features[name] = val

# Predict
if st.button("Predict Quality"):
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    result = "Good (Score ≥ 6)" if pred == 1 else "Poor (Score < 6)"
    st.success(f"**{result}**")
    st.write(f"**Confidence: {prob[pred]:.1%}**")

    # Show top features
    st.caption("Top predictors: alcohol, volatile acidity, sulphates")