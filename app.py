import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, abort

app = Flask(__name__)

# ==== Load Model & Encoders ====
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "career_model.joblib"
INTEREST_ENC_PATH = MODELS_DIR / "interest_encoder.joblib"
TARGET_ENC_PATH = MODELS_DIR / "target_encoder.joblib"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"

if not (MODEL_PATH.exists() and INTEREST_ENC_PATH.exists() and TARGET_ENC_PATH.exists() and FEATURE_ORDER_PATH.exists()):
    raise RuntimeError("Model artifacts missing. Run:  python train.py")

model = joblib.load(MODEL_PATH)
interest_le = joblib.load(INTEREST_ENC_PATH)
target_le = joblib.load(TARGET_ENC_PATH)
feature_order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))

# ==== DB ====
DB_PATH = Path("career.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==== Routes ====
@app.route("/", methods=["GET"])
def index():
    # Allowed interests = classes seen during training
    allowed_interests = list(interest_le.classes_)
    return render_template("index.html", allowed_interests=allowed_interests)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("name", "").strip()
        english = request.form.get("English", "").strip()
        math = request.form.get("Math", "").strip()
        science = request.form.get("Science", "").strip()
        history = request.form.get("History", "").strip()
        geography = request.form.get("Geography", "").strip()
        interest = request.form.get("Interest", "").strip()

        # Basic validation
        if not name:
            abort(400, "Name is required.")
        for fld, val in [("English", english), ("Math", math), ("Science", science),
                         ("History", history), ("Geography", geography), ("Interest", interest)]:
            if val == "":
                abort(400, f"{fld} is required.")

        # Coerce numeric marks
        try:
            marks = {
                "English": float(english),
                "Math": float(math),
                "Science": float(science),
                "History": float(history),
                "Geography": float(geography),
            }
        except ValueError:
            abort(400, "All subject marks must be numeric (e.g., 75 or 75.0).")

        # Ensure interest is in known set; fallback safely to "Other" if unseen
        if interest not in interest_le.classes_:
            if "Other" in interest_le.classes_:
                interest = "Other"
            else:
                # Extreme fallback: map to first known class
                interest = interest_le.classes_[0]

        # Build input DataFrame in the SAME order as training
        row = {
            "English": marks["English"],
            "Math": marks["Math"],
            "Science": marks["Science"],
            "History": marks["History"],
            "Geography": marks["Geography"],
            "Interest": interest,
        }
        df_in = pd.DataFrame([row])
        # Encode interest
        df_in["Interest"] = interest_le.transform(df_in["Interest"])

        # Reorder columns exactly
        df_in = df_in[feature_order]

        # Predict
        probs = getattr(model, "predict_proba", None)
        y_pred_enc = model.predict(df_in)[0]
        career = target_le.inverse_transform([y_pred_enc])[0]
        confidence = float(np.max(probs(df_in)[0]) * 100.0) if probs else 0.0

        # Store in DB
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO students (name, english, math, science, history, geography, interest)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, marks["English"], marks["Math"], marks["Science"], marks["History"], marks["Geography"], interest),
            )
            student_id = cur.lastrowid
            cur.execute(
                """
                INSERT INTO predictions (student_id, predicted_career, confidence)
                VALUES (?, ?, ?)
                """,
                (student_id, career, confidence),
            )
            conn.commit()

        return render_template("result.html", name=name, career=career, confidence=f"{confidence:.2f}%")

    except Exception as e:
        # Show friendly error
        return render_template("result.html", name="Error", career=f"❌ {type(e).__name__}", confidence=str(e)), 400

@app.route("/admin", methods=["GET"])
def admin():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.id, s.name, s.english, s.math, s.science, s.history, s.geography, s.interest,
                   p.predicted_career, p.confidence, p.created_at
            FROM students s
            JOIN predictions p ON p.student_id = s.id
            ORDER BY p.created_at DESC
            """
        )
        rows = cur.fetchall()
    return render_template("admin.html", rows=rows)

if __name__ == "__main__":
    # Ensure DB exists (run db_init.py once before first run)
    if not DB_PATH.exists():
        raise RuntimeError("SQLite DB not found. Run:  python db_init.py")
    app.run(debug=True)
