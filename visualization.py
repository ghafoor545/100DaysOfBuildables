import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Genuine", "Fake"], y=["Genuine", "Fake"],
        colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True
    ))
    fig.update_layout(title="Confusion Matrix (Test Set)", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_feature_importance(model, model_name, feature_names):
    if model_name == "Logistic Regression":
        coef = model.named_steps["clf"].coef_[0]
        importance_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
        fig = px.bar(importance_df, x="Feature", y="Coefficient", title="Feature Importance")
    elif model_name == "Decision Tree":
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
    else:  # KNN
        fig = None  # No feature importance for KNN
    return fig