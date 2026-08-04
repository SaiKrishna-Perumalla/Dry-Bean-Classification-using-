import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")
st.title("🫘 Dry Bean Classification using ANN")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = load_model(MODEL_DIR / r"C:\Users\saive\OneDrive\Documents\python 461\ANN\dry_bean_ann_model.keras")
scaler = joblib.load(MODEL_DIR / r"C:\Users\saive\OneDrive\Documents\python 461\ANN\dry_bean_scaler.pkl")
encoder = joblib.load(MODEL_DIR / r"C:\Users\saive\OneDrive\Documents\python 461\ANN\dry_bean_label_encoder.pkl")

features = [
"Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRation",
"Eccentricity","ConvexArea","EquivDiameter","Extent","Solidity",
"roundness","Compactness","ShapeFactor1","ShapeFactor2","ShapeFactor3","ShapeFactor4"
]

st.sidebar.header("Input Features")
vals = [st.sidebar.number_input(f, value=0.0, format="%.4f") for f in features]

if st.button("Predict"):
    df = pd.DataFrame([vals], columns=features)
    x = scaler.transform(df)
    pred = model.predict(x)
    idx = np.argmax(pred, axis=1)
    bean = encoder.inverse_transform(idx)[0]
    st.success(f"Prediction: {bean}")
    st.write("Confidence:", float(np.max(pred))*100)
