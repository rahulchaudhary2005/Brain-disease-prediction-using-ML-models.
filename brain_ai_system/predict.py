import joblib
import pandas as pd

# Load model & encoder
model = joblib.load("./model/brain_disease_model.pkl")
label_encoder = joblib.load("./model/label_encoder.pkl")

def predict_brain_disease(symptoms):
    df = pd.DataFrame([symptoms])
    pred = model.predict(df)[0]
    disease = label_encoder.inverse_transform([pred])[0]
    return disease

# Example
if __name__ == "__main__":
    user_input = {
        "headache": 1,
        "nausea": 1,
        "vomiting": 0,
        "dizziness": 1,
        "altered_sensorium": 0,
        "loss_of_balance": 1,
        "seizures": 0,
        "visual_disturbances": 1,
        "coma": 0,
        "muscle_weakness": 0
    }

    result = predict_brain_disease(user_input)
    print("Predicted Brain Disease:", result)
