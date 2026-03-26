import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

# -----------------------------
# Base Path (IMPORTANT for deployment)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

# -----------------------------
# Load Model & Preprocessors (Cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_assets():
    model = load_model(os.path.join(BASE_DIR, "Titanic_ann_model_v1.keras"))

    with open(os.path.join(BASE_DIR, "sex_label_encoder_v1.pkl"), "rb") as f:
        sex_encoder = pickle.load(f)

    with open(os.path.join(BASE_DIR, "embarked_ohe_encoder_v1.pkl"), "rb") as f:
        embarked_encoder = pickle.load(f)

    with open(os.path.join(BASE_DIR, "feature_scaler_v1.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, "model_feature_columns_v1.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    return model, sex_encoder, embarked_encoder, scaler, feature_columns


try:
    model, sex_encoder, embarked_encoder, scaler, feature_columns = load_assets()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", 0, 100, 25)

with col2:
    sibsp = st.number_input("SibSp", 0, 10, 0)
    parch = st.number_input("Parch", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)

embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
embarked_choice = st.selectbox("Embarked", list(embarked_map.keys()))
embarked = embarked_map[embarked_choice]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    try:
        # -----------------------------
        # 1. Create Input Data
        # -----------------------------
        input_df = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked]
        })

        # -----------------------------
        # 2. Encode Sex
        # -----------------------------
        if sex not in sex_encoder.classes_:
            raise ValueError(f"Unexpected Sex value: {sex}")

        input_df["Sex"] = sex_encoder.transform(input_df["Sex"])

        # -----------------------------
        # 3. OneHot Encode Embarked
        # -----------------------------
        embarked_encoded = embarked_encoder.transform(input_df[["Embarked"]])

        # Convert sparse → dense (IMPORTANT)
        if hasattr(embarked_encoded, "toarray"):
            embarked_encoded = embarked_encoded.toarray()

        embarked_df = pd.DataFrame(
            embarked_encoded,
            columns=embarked_encoder.get_feature_names_out(["Embarked"])
        )

        # Merge
        input_df = input_df.drop("Embarked", axis=1)
        input_df = pd.concat([input_df.reset_index(drop=True), embarked_df], axis=1)

        # -----------------------------
        # 4. Column Alignment
        # -----------------------------
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        # -----------------------------
        # 5. Scaling
        # -----------------------------
        input_scaled = scaler.transform(input_df)

        # -----------------------------
        # 6. Prediction
        # -----------------------------
        prediction = model.predict(input_scaled)
        prob = float(prediction.flatten()[0])

        # -----------------------------
        # 7. Output
        # -----------------------------
        st.divider()

        if prob > 0.5:
            st.success("✅ Survived")
            st.write(f"Confidence: **{prob*100:.2f}%**")
        else:
            st.error("❌ Did Not Survive")
            st.write(f"Confidence: **{(1-prob)*100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.info("Check encoders, scaler, and feature columns consistency.")