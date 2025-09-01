import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = Path("data/career_data.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Expected columns
expected_cols = ["English", "Math", "Science", "History", "Geography", "Interest", "career_path"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing expected columns: {missing}")

# Separate features/target
feature_cols = ["English", "Math", "Science", "History", "Geography", "Interest"]
target_col = "career_path"

X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode categorical columns
interest_le = LabelEncoder()
X["Interest"] = interest_le.fit_transform(X["Interest"])

# Encode target
target_le = LabelEncoder()
y_enc = target_le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Model
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model trained. Holdout accuracy: {acc:.3f}")

# Save artifacts
joblib.dump(model, MODELS_DIR / "career_model.joblib")
joblib.dump(interest_le, MODELS_DIR / "interest_encoder.joblib")
joblib.dump(target_le, MODELS_DIR / "target_encoder.joblib")

# Save the exact feature order used during fit
feature_order = list(X.columns)  # ensures ["English","Math","Science","History","Geography","Interest"]
with open(MODELS_DIR / "feature_order.json", "w", encoding="utf-8") as f:
    json.dump(feature_order, f, ensure_ascii=False, indent=2)

print("✅ Saved model & encoders to models/")
