from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import __main__
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps[col]).fillna(0)
        return X

__main__.FrequencyEncoder = FrequencyEncoder

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "NoteBooks" / "NoteBooks" / "ML_Models" / "best_xgboost.pkl"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "used_cars.csv"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def get_select_options(df: pd.DataFrame):
    return {
        "brands": sorted(df["brand"].dropna().unique().tolist()),
        "fuel_types": sorted(df["fuel_type"].dropna().unique().tolist()),
        "transmissions": sorted(df["transmission"].dropna().unique().tolist()),
        "colors": sorted(df["ext_col"].dropna().unique().tolist()),
        "interior_colors": sorted(df["int_col"].dropna().unique().tolist()),
    }


def format_price(value: float) -> str:
    return f"${value:,.0f}"

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")

st.title("Used Car Price Estimator")
st.caption("Predict a used car price using the trained XGBoost pipeline from the notebook.")

try:
    model = load_model()
    reference_df = load_reference_data()
    options = get_select_options(reference_df)
except Exception as exc:
    st.error(f"The app could not start: {exc}")
    st.stop()

with st.form("car_price_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", options["brands"])
        model_name = st.text_input("Model", value="Corolla")
        model_year = st.number_input("Model year", min_value=1990, max_value=2035, value=2020, step=1)
        milage = st.number_input("Mileage (mi)", min_value=0, max_value=500000, value=35000, step=100)
        fuel_type = st.selectbox("Fuel type", options["fuel_types"])
        transmission = st.selectbox("Transmission", options["transmissions"])

    with col2:
        ext_col = st.selectbox("Exterior color", options["colors"])
        int_col = st.selectbox("Interior color", options["interior_colors"])
        accident = st.selectbox(
            "Accident status",
            ["None reported", "At least 1 accident or damage reported", "Unknown"],
        )
        clean_title = st.selectbox("Clean title", ["Yes", "Unknown"])
        engine_capacity = st.number_input("Engine capacity (L)", min_value=0.0, max_value=10.0, value=1.8, step=0.1)
        horse_power = st.number_input("Horse power (HP)", min_value=0, max_value=1000, value=140, step=10)

    submitted = st.form_submit_button("Estimate price")

if submitted:
    accident_occurred = "NO" if accident == "None reported" else "YES"

    payload = pd.DataFrame([
        {
            "brand": brand,
            "model": model_name,
            "model_year": int(model_year),
            "milage": float(milage),
            "fuel_type": fuel_type,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident,
            "clean_title": clean_title,
            "Accident_Occured": accident_occurred,
            "Engine_Capacity": float(engine_capacity),
            "Horse_Power": float(horse_power),
        }
    ])

    prediction = float(model.predict(payload)[0])
    st.success("Estimated market value")
    st.metric(label="Predicted car price", value=format_price(prediction))

    with st.expander("Input preview"):
        st.dataframe(payload, use_container_width=True)