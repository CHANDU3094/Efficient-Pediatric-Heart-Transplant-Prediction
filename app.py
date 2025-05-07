import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# Load model and data
rf_as_best_model = joblib.load("rf_as_best_model.pkl")
icu_df = pd.read_csv("SDP_15_F.csv")

# Filter only patients who survived
survived_patients = icu_df[icu_df["hospital_expire_flag"] == 0].copy()
drop_columns = ["Unnamed: 0", "hospital_expire_flag"]
survived_patients = survived_patients.drop(columns=drop_columns, errors="ignore")

# Standardize data
scaler = StandardScaler()
survived_patients_scaled = pd.DataFrame(
    scaler.fit_transform(survived_patients),
    columns=survived_patients.columns
)

# Streamlit UI
st.title("🫀 Pediatric Heart Transplant Outcome Predictor")

st.markdown("### Enter Patient Data")

# Inputs
admission_age = st.number_input("Admission Age", min_value=0.0)
heartrate_mean = st.number_input("Heart Rate Mean", min_value=0.0)
meanbp_mean = st.number_input("Mean Blood Pressure", min_value=0.0)
spo2_mean = st.number_input("SpO2 Mean", min_value=0.0)
glucose_mean = st.number_input("Glucose Mean", min_value=0.0)
lactate_max = st.number_input("Lactate Max", min_value=0.0)
aniongap_max = st.number_input("Anion Gap Max", min_value=0.0)
creatinine_max = st.number_input("Creatinine Max", min_value=0.0)
bun_max = st.number_input("BUN Max", min_value=0.0)
wbc_max = st.number_input("WBC Max", min_value=0.0)
mingcs = st.number_input("Minimum GCS", min_value=0.0)
BMI = st.number_input("BMI", min_value=0.0)
respxvent = st.number_input("Resp x Vent", min_value=0.0)
gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
intubated = st.selectbox("Intubated (0: No, 1: Yes)", [0, 1])

if st.button("Predict"):
    user_data = {
        "admission_age": admission_age,
        "heartrate_mean": heartrate_mean,
        "meanbp_mean": meanbp_mean,
        "spo2_mean": spo2_mean,
        "glucose_mean": glucose_mean,
        "lactate_max": lactate_max,
        "aniongap_max": aniongap_max,
        "creatinine_max": creatinine_max,
        "bun_max": bun_max,
        "wbc_max": wbc_max,
        "mingcs": mingcs,
        "BMI": BMI,
        "respxvent": respxvent,
        "gender": gender,
        "intubated": intubated
    }

    # Ensure consistent columns
    user_df = pd.DataFrame([user_data])
    user_df = user_df[survived_patients.columns]
    user_df.fillna(survived_patients.median(), inplace=True)

    # Prediction
    prediction = rf_as_best_model.predict(user_df)[0]
    probability = rf_as_best_model.predict_proba(user_df)[:, 1][0]  # Death

    # Find best donor
    recipient_scaled = scaler.transform(user_df)
    distances = survived_patients_scaled.apply(lambda row: euclidean(row, recipient_scaled[0]), axis=1)
    best_index = distances.idxmin()
    best_donor = survived_patients.iloc[best_index]
    similarity_score = distances.min()

    # Output
    risk_label = "Die" if prediction == 1 else "Survive"

    st.markdown(f"### 📌 **Patient will** {'🟥 Die' if prediction == 1 else '🟩 Survive'}")
    st.markdown(f"**Probability of Death:** {probability:.4f}")
    st.markdown(f"**Probability of Survival:** {1 - probability:.4f}")

    st.markdown("---")
    st.markdown("### 👤 **Best Donor Found:**")
    st.dataframe(best_donor)

    st.markdown(f"**Similarity Score:** {similarity_score:.4f}")
