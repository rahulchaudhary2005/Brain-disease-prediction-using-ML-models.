import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(csv_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Heart_desease_prdition.csv"))

    df = pd.read_csv(DATA_PATH)

    # Drop ID
    df.drop(columns=["id"], inplace=True)

    # Fill missing values
    num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
    for col in num_cols:
       df[col]= df[col].fillna(df[col].median())

    cat_cols = ["fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in cat_cols:
       df[col]= df[col].fillna(df[col].mode()[0])
      # 🔥 FINAL SAFETY CHECK
    #df = df.infer_objects(copy=False) 

    # Binary target
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    # Encode categorical
    encoder = LabelEncoder()
    cat_features = ["sex", "dataset", "cp", "restecg", "slope", "thal"]
    for col in cat_features:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop(columns=["num"])
    y = df["num"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, list(X.columns)
