import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Dataset & Model Insights",page_icon="heart.png", layout="centered")

model = joblib.load("heart_disease_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction App")
st.write("Provide the patient data and predict presence of heart disease")


age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ['typical angina','asymptomatic','non-anginal','atypical angina'])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Cholesterol (chol)", 100, 500, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [True, False])
thalch = st.number_input("Max Heart Rate (thalch)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [True, False])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
ca = st.number_input("Number of major vessels (ca)", 0, 3, 0)
slope = st.selectbox("Slope", ['downsloping','flat','upsloping'])
thal = st.selectbox("Thal", ['fixed defect','normal','reversable defect'])
restecg = st.selectbox("Resting ECG (restecg)", ['lv hypertrophy','normal','st-t abnormality'])


input_dict = {
    "age": age,
    "sex": 0 if sex == "Male" else 1,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs else 0,
    "thalch": thalch,
    "exang": 1 if exang else 0,
    "oldpeak": oldpeak,
    "ca": ca,
    "restecg": restecg,
    "cp": cp,
    "slope": slope,
    "thal": thal,
}

df = pd.DataFrame([input_dict])
df['ca_missing'] = df['ca'].isnull().astype(int)
df['thal_missing'] = df['thal'].isnull().astype(int)
df['slope_missing'] = df['slope'].isnull().astype(int)

enc = encoder.transform(df[['cp','restecg','slope','thal']])
encoded_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(['cp','restecg','slope','thal']))
encoded_df = pd.concat([df.drop(columns=['cp','restecg','slope','thal']), encoded_df],axis=1)

num_columns = ['age','trestbps','chol','thalch','oldpeak','ca']
encoded_df[num_columns] = scaler.transform(encoded_df[num_columns])


if st.button("Predict"):
    pred = model.predict(encoded_df)[0]
    prob = float(model.predict_proba(encoded_df)[0][1]) * 100

    if pred == 1:
        st.error(f"Heart Disease Detected — Probability: {prob:.2f}%")
    else:
        st.success(f"No Heart Disease — Probability: {prob:.2f}%")
