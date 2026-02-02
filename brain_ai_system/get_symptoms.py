import joblib

model = joblib.load("model/brain_disease_model.pkl")

symptoms = model.get_booster().feature_names

print("Total symptoms:", len(symptoms))
print(symptoms)
