import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("kidney_disease.csv")

# Drop ID
df.drop(columns=["id"], inplace=True)

# Encode target
df["classification"] = df["classification"].map({"ckd":1, "notckd":0})

# Encode categorical columns
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Handle missing values
imputer = SimpleImputer(strategy="median")
df[:] = imputer.fit_transform(df)

# Split
X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model (HIGH ACCURACY)
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save for AIBoot
joblib.dump(model, "ckd_model.pkl")
joblib.dump(imputer, "ckd_imputer.pkl")
joblib.dump(X.columns.tolist(), "ckd_features.pkl")
