import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

st.title("💳 Credit Card Fraud Detection")
st.write("Using Machine Learning (Random Forest)")

# Sample or real dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/opengeekslab/opengeekslab.github.io/main/datasets/creditcard.csv")
    return df

df = load_data()

# Preprocessing
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
st.write("### Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

# Try your own input
st.write("### Try a New Transaction")

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    input_data.append(val)

if st.button("Predict"):
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")