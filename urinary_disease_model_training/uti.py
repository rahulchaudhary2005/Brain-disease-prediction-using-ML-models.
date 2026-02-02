import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("./urinalysis_tests.xlsx")

df.drop(columns=["Unnamed: 0"], inplace=True)

# Encode all categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

# Target
y = df["Diagnosis"].map({"NEGATIVE":0, "POSITIVE":1})
X = df.drop("Diagnosis", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "uti_model.pkl")
joblib.dump(X.columns.tolist(), "uti_features.pkl")

