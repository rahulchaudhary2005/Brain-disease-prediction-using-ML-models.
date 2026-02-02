from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="AIBoot Unified Medical Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ===================== LOAD MODELS =====================
ckd_model = joblib.load(os.path.join(MODEL_DIR, "ckd_model.pkl"))
ckd_imputer = joblib.load(os.path.join(MODEL_DIR, "ckd_imputer.pkl"))
ckd_features = joblib.load(os.path.join(MODEL_DIR, "ckd_features.pkl"))

ckd_lifestyle_model = joblib.load(os.path.join(MODEL_DIR, "ckd_lifestyle_model.pkl"))
ckd_lifestyle_features = joblib.load(os.path.join(MODEL_DIR, "ckd_lifestyle_features.pkl"))

uti_model = joblib.load(os.path.join(MODEL_DIR, "uti_model.pkl"))
uti_features = joblib.load(os.path.join(MODEL_DIR, "uti_model.pkl"))

# ===================== HELPER FUNCTIONS =====================
def calculate_risk(prob):
    return round(prob * 100, 2)

def medical_advice(disease, risk):
    if risk < 30:
        return "Low risk. Maintain healthy lifestyle and hydration."
    elif risk < 60:
        return "Moderate risk. Lifestyle changes and monitoring recommended."
    else:
        return f"High risk of {disease}. Immediate doctor consultation advised."

def doctor_consult(risk):
    return risk >= 60

# ===================== CKD PREDICTION =====================
@app.post("/predict/ckd")
def predict_ckd(data: dict):
    df = pd.DataFrame([data])
    df = df[ckd_features]
    df[:] = ckd_imputer.transform(df)

    prob = ckd_model.predict_proba(df)[0][1]
    risk = calculate_risk(prob)

    return {
        "disease": "Chronic Kidney Disease",
        "risk_percentage": risk,
        "advice": medical_advice("CKD", risk),
        "doctor_consultation_required": doctor_consult(risk)
    }

# ===================== LIFESTYLE CKD =====================
@app.post("/predict/ckd-lifestyle")
def predict_ckd_lifestyle(data: dict):
    df = pd.DataFrame([data])
    df = df[ckd_lifestyle_features]

    prob = ckd_lifestyle_model.predict_proba(df)[0][1]
    risk = calculate_risk(prob)

    return {
        "disease": "Lifestyle-related CKD",
        "risk_percentage": risk,
        "advice": medical_advice("CKD", risk),
        "doctor_consultation_required": doctor_consult(risk)
    }

# ===================== UTI =====================
@app.post("/predict/uti")
def predict_uti(data: dict):
    df = pd.DataFrame([data])
    df = df[uti_features]

    prob = uti_model.predict_proba(df)[0][1]
    risk = calculate_risk(prob)

    return {
        "disease": "Urinary Tract Infection",
        "risk_percentage": risk,
        "advice": medical_advice("UTI", risk),
        "doctor_consultation_required": doctor_consult(risk)
    }
