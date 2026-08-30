
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# =========================================================
# Title
# =========================================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's information below to predict "
    "the likelihood of heart disease."
)


# =========================================================
# Load Dataset
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("heart_disease_uci.csv")

    # Create binary target
    df["target"] = df["num"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Fill missing values exactly as training
    df[["fbs", "exang"]] = df[["fbs", "exang"]].fillna(
        df[["fbs", "exang"]].mode().iloc[0]
    ).astype(int)

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=[
            "cp",
            "dataset",
            "restecg",
            "thal",
            "slope",
            "sex"
        ],
        drop_first=True,
        dtype=int
    )

    # Remove original target column
    df = df.drop("num", axis=1)

    return df


df = load_data()


# =========================================================
# Load Preprocessing Objects and Model
# =========================================================

imputer = joblib.load("imputer.pkl")
kbest = joblib.load("kbest.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("voting_model.pkl")


# =========================================================
# Get Feature Names Used During Training
# =========================================================

feature_columns = df.drop("target", axis=1).columns


# =========================================================
# User Inputs
# =========================================================

st.subheader("Patient Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )


with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "lv hypertrophy",
            "st-t abnormality"
        ]
    )

    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


# =========================================================
# Additional Inputs
# =========================================================

st.subheader("Additional Information")


col3, col4 = st.columns(2)


with col3:

    ca = st.number_input(
        "Number of Major Vessels (CA)",
        min_value=0,
        max_value=4,
        value=0
    )

    slope = st.selectbox(
        "Slope",
        [
            "upsloping",
            "flat",
            "downsloping"
        ]
    )


with col4:

    dataset = st.selectbox(
        "Dataset",
        [
            "Cleveland",
            "Hungary",
            "Switzerland",
            "VA Long Beach"
        ]
    )


# =========================================================
# Prediction
# =========================================================

if st.button("🔍 Predict", use_container_width=True):

    # -----------------------------------------------------
    # Create empty row with EXACT training columns
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )


    # -----------------------------------------------------
    # Numerical Features
    # -----------------------------------------------------

    if "id" in input_data.columns:
        input_data["id"] = 0

    if "age" in input_data.columns:
        input_data["age"] = age

    if "trestbps" in input_data.columns:
        input_data["trestbps"] = trestbps

    if "chol" in input_data.columns:
        input_data["chol"] = chol

    if "fbs" in input_data.columns:
        input_data["fbs"] = fbs

    if "thalch" in input_data.columns:
        input_data["thalch"] = thalch

    if "exang" in input_data.columns:
        input_data["exang"] = exang

    if "oldpeak" in input_data.columns:
        input_data["oldpeak"] = oldpeak

    if "ca" in input_data.columns:
        input_data["ca"] = ca


    # -----------------------------------------------------
    # One-Hot Encoded Features
    # -----------------------------------------------------

    # Sex
    if sex == "Male":

        if "sex_Male" in input_data.columns:
            input_data["sex_Male"] = 1


    # Chest Pain
    cp_column = f"cp_{cp}"

    if cp_column in input_data.columns:
        input_data[cp_column] = 1


    # Dataset
    dataset_column = f"dataset_{dataset}"

    if dataset_column in input_data.columns:
        input_data[dataset_column] = 1


    # Resting ECG
    restecg_column = f"restecg_{restecg}"

    if restecg_column in input_data.columns:
        input_data[restecg_column] = 1


    # Slope
    slope_column = f"slope_{slope}"

    if slope_column in input_data.columns:
        input_data[slope_column] = 1


    # -----------------------------------------------------
    # Replace possible missing values
    # -----------------------------------------------------

    input_data = input_data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # -----------------------------------------------------
    # Imputation
    # -----------------------------------------------------

    input_imputed = imputer.transform(
        input_data
    )


    # -----------------------------------------------------
    # SelectKBest
    # -----------------------------------------------------

    input_selected = kbest.transform(
        input_imputed
    )


    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    input_scaled = scaler.transform(
        input_selected
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        input_scaled
    )


    probability = model.predict_proba(
        input_scaled
    )[0][1]


    # -----------------------------------------------------
    # Display Result
    # -----------------------------------------------------

    st.divider()

    if prediction[0] == 1:

        st.error(
            "❤️ Heart Disease Detected"
        )

    else:

        st.success(
            "💚 No Heart Disease Detected"
        )


    st.write(
        f"### Probability of Heart Disease: "
        f"{probability * 100:.2f}%"
    )
