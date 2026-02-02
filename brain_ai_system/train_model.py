import os
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# ---------------------------------
# Setup
# ---------------------------------
os.makedirs("model", exist_ok=True)

# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv("./data/dataset.csv")
print("✅ Dataset loaded:", df.shape)

# ---------------------------------
# Identify columns
# ---------------------------------
TARGET_COL = "Disease"
SYMPTOM_COLS = [col for col in df.columns if col != TARGET_COL]

# ---------------------------------
# Collect ALL unique symptoms
# ---------------------------------
all_symptoms = set()

for col in SYMPTOM_COLS:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted(all_symptoms)
print(f"🧠 Total unique symptoms: {len(all_symptoms)}")

# ---------------------------------
# Create binary symptom matrix
# ---------------------------------
encoded_rows = []

for _, row in df.iterrows():
    symptom_vector = {symptom: 0 for symptom in all_symptoms}

    for col in SYMPTOM_COLS:
        symptom = row[col]
        if pd.notna(symptom):
            symptom_vector[symptom] = 1

    symptom_vector[TARGET_COL] = row[TARGET_COL]
    encoded_rows.append(symptom_vector)

encoded_df = pd.DataFrame(encoded_rows)
print("✅ Encoded dataset shape:", encoded_df.shape)

# ---------------------------------
# Encode disease labels
# ---------------------------------
le = LabelEncoder()
encoded_df[TARGET_COL] = le.fit_transform(encoded_df[TARGET_COL])

joblib.dump(le, "model/label_encoder.pkl")

print("🧠 Disease classes:", list(le.classes_))

# ---------------------------------
# Train-test split
# ---------------------------------
X = encoded_df.drop(TARGET_COL, axis=1)
y = encoded_df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------
# Train XGBoost (FINAL SAFE CONFIG)
# ---------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(le.classes_),
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------
# Evaluation
# ---------------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------
# Save model
# ---------------------------------
joblib.dump(model, "model/brain_disease_model.pkl")

print("\n🎉 SUCCESS: Model trained correctly!")
