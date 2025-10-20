# Banknote Authenticator (Streamlit)

**Predict genuine vs. fake banknotes** using 3 models in a single Streamlit app.

## Models
- **Logistic Regression** (baseline, ~0.99 acc)
- **Decision Tree** (`max_depth=5`)
- **KNN** (`n_neighbors=5`)

## Features
- UCI dataset (~1372 rows) loaded online  
- Sidebar **model selector** (`st.selectbox`)  
- Train/test **accuracy, F1, MAE** table  
- **Confusion matrix** + **feature importance** (Plotly)  
- **Live prediction** UI

## Run in PyCharm
```bash
pip install -r requirements.txt
streamlit run app.py