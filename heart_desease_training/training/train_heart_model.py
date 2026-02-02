import json
import os
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess
from model_selection   import select_best_model


# Load & preprocess
X, y, scaler, feature_order = load_and_preprocess(None)

# -------------------------------
# FINAL NaN CHECK (CRITICAL)
# -------------------------------
# if df.isnull().sum().sum() > 0:
#     print("❌ NaNs still present:")
#     print(df.isnull().sum())
#     raise ValueError("NaN values detected after preprocessing")


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train & select best model
model, model_name, accuracy = select_best_model(
    X_train, X_test, y_train, y_test
)


ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model_path = os.path.join(ARTIFACTS_DIR, "heart_model01.pkl")
scaler_path = os.path.join(ARTIFACTS_DIR, "heart_scaler.pkl")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print("✅ Model saved at:", model_path)
print("✅ Scaler saved at:", scaler_path)
# Save artifacts
#joblib.dump(model, "../artifacts/heart_model01.pkl")
joblib.dump(scaler, "../artifacts/heart_scaler01.pkl")

with open("../artifacts/feature_order.json", "w") as f:
    json.dump(feature_order, f)

with open("../artifacts/model_info.json", "w") as f:
    json.dump(
        {"model": model_name, "accuracy": accuracy},
        f,
        indent=4
    )

print("✅ Heart disease model trained")
print("Best Model:", model_name)
print("Accuracy:", accuracy)
