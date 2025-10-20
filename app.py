import streamlit as st
from data_loader import load_and_preprocess_data
from models import get_model, train_model
from evaluation import evaluate_model
from visualization import plot_confusion_matrix, plot_feature_importance

# Page config
st.set_page_config(page_title="Banknote Authentication", layout="wide")

# Title
st.title("Banknote Authentication Model Selector")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN"],
    help="Select a model to train and evaluate."
)

# Load and preprocess data
try:
    df, X_train, X_test, y_train, y_test = load_and_preprocess_data()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Data exploration
st.header("Data Exploration")
st.write("First 5 rows:")
st.dataframe(df.head())
st.write("Data types:")
st.write(df.dtypes)
st.write("Missing values:")
st.write(df.isnull().sum())
st.write("Dataset shape:", df.shape)

# Train model
try:
    model = get_model(model_choice)
    model = train_model(model, X_train, y_train)
except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

# Evaluate model
metrics_df, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

# Display metrics
st.subheader("Model Performance")
st.table(metrics_df)

# Visualizations
st.subheader("Visualizations")
fig_cm = plot_confusion_matrix(y_test, y_test_pred)
st.plotly_chart(fig_cm)

if model_choice in ["Logistic Regression", "Decision Tree"]:
    fig_importance = plot_feature_importance(model, model_choice, X_train.columns)
    if fig_importance:
        st.plotly_chart(fig_importance)

# Prediction interface
st.header("Predict a Banknote")
st.write("Enter feature values to predict if a banknote is genuine (0) or fake (1).")
variance = st.number_input("Variance", value=0.0, step=0.1)
skewness = st.number_input("Skewness", value=0.0, step=0.1)
curtosis = st.number_input("Curtosis", value=0.0, step=0.1)
entropy = st.number_input("Entropy", value=0.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    try:
        prediction = model.predict(input_data)[0]
        label = "Genuine" if prediction == 0 else "Fake"
        st.success(f"Prediction: {label} (Class {prediction})")
    except Exception as e:
        st.error(f"Prediction error: {e}")