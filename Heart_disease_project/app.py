import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction App")
st.write("Provide the following details to predict the chance of heart disease.")

# Input fields matching the dataset columns
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (0–3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=700, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("Yes", 1), ("No", 0)])
restecg = st.selectbox("Resting ECG (0–2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[("Yes", 1), ("No", 0)])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope (0–2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", options=[1, 2, 3])

if st.button("Predict"):
    # Extract numeric values from dropdowns with (label, value)
    sex_val = sex[1] if isinstance(sex, tuple) else sex
    fbs_val = fbs[1] if isinstance(fbs, tuple) else fbs
    exang_val = exang[1] if isinstance(exang, tuple) else exang

    input_data = np.array([[
        age,
        sex_val,
        cp,
        trestbps,
        chol,
        fbs_val,
        restecg,
        thalach,
        exang_val,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("High chance of heart disease.")
    else:
        st.success("Low chance of heart disease.")