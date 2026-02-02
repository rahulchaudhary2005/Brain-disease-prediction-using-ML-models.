from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ---------------------------------
# Load trained model & encoder
# ---------------------------------
model = joblib.load("model/brain_disease_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# ---------------------------------
# Brain / Head diseases to show
# ---------------------------------
BRAIN_DISEASES = [
    "Migraine",
    "Paralysis (brain hemorrhage)",
    "(vertigo) Paroymsal  Positional Vertigo",
    "Cervical spondylosis"
]

# ---------------------------------
# Disease → Symptom explanation map
# ---------------------------------
DISEASE_SYMPTOM_MAP = {
    "Paralysis (brain hemorrhage)": [
        "loss_of_balance",
        "vomiting",
        "altered_sensorium",
        "headache",
        "weakness_of_one_body_side"
    ],
    "(vertigo) Paroymsal  Positional Vertigo": [
        "dizziness",
        "loss_of_balance",
        "nausea",
        "vomiting"
    ],
    "Migraine": [
        "headache",
        "nausea",
        "vomiting",
        "visual_disturbances"
    ],
    "Cervical spondylosis": [
        "neck_pain",
        "headache",
        "dizziness",
        "muscle_weakness"
    ]
}

# ---------------------------------
# FastAPI app
# ---------------------------------
app = FastAPI(
    title="AI Boot – Brain & Head Disease Predictor",
    description="Symptom-based AI risk prediction (not a medical diagnosis)",
    version="1.0"
)

# ---------------------------------
# CORS (for frontend)
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# Input schema
# ---------------------------------
class SymptomInput(BaseModel):
    symptoms: list[str]

# ---------------------------------
# Encode symptoms
# ---------------------------------
def encode_user_symptoms(user_symptoms, all_symptoms):
    encoded = dict.fromkeys(all_symptoms, 0)
    for s in user_symptoms:
        if s in encoded:
            encoded[s] = 1
    return encoded

# ---------------------------------
# Explain prediction
# ---------------------------------
def explain_prediction(disease, user_symptoms):
    expected = DISEASE_SYMPTOM_MAP.get(disease, [])
    matched = list(set(expected) & set(user_symptoms))

    if not matched:
        return (
            "The prediction is based on the overall symptom pattern similarity "
            "observed in the trained data."
        )

    return (
        f"The prediction is influenced by the presence of symptoms such as "
        f"{', '.join(matched)}, which are commonly associated with {disease}."
    )

# ---------------------------------
# Doctor recommendation logic
# ---------------------------------
def recommend_doctor(disease, probability):
    if disease == "Paralysis (brain hemorrhage)" and probability >= 25:
        return "Emergency (ER) – Seek immediate medical attention"

    if disease in [
        "Migraine",
        "(vertigo) Paroymsal  Positional Vertigo",
        "Cervical spondylosis"
    ]:
        return "Neurologist"

    return "General Physician (GP)"

# ---------------------------------
# Prediction endpoint
# ---------------------------------
@app.post("/predict")
def predict_brain_disease(data: SymptomInput):

    # Get model feature names
    all_symptoms = model.get_booster().feature_names

    # Encode user input
    encoded_input = encode_user_symptoms(data.symptoms, all_symptoms)
    X = pd.DataFrame([encoded_input])

    # Predict probabilities
    probs = model.predict_proba(X)[0]
    diseases = label_encoder.inverse_transform(
        np.arange(len(probs))
    )

    # Filter brain diseases only
    filtered = {
        diseases[i]: float(probs[i])   # numpy → python float
        for i in range(len(diseases))
        if diseases[i] in BRAIN_DISEASES
    }

    # Normalize probabilities to 100%
    total = sum(filtered.values())
    normalized = {
        k: round((v / total) * 100, 2)
        for k, v in filtered.items()
    }

    # Sort by highest risk
    normalized = dict(
        sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    )

    # Top disease
    top_disease = list(normalized.keys())[0]
    top_probability = normalized[top_disease]

    # Explanation & doctor suggestion
    explanation = explain_prediction(top_disease, data.symptoms)
    doctor = recommend_doctor(top_disease, top_probability)

    return {
        "most_likely_disease": top_disease,
        "risk_probabilities": normalized,
        "why_this_result": explanation,
        "recommended_doctor": doctor,
        "disclaimer": (
            "This is an AI-based risk prediction system for educational purposes. "
            "It is NOT a medical diagnosis. Please consult a qualified doctor."
        )
    }
