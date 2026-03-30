from flask import Flask, render_template, request, flash, send_file, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import mysql.connector
try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not installed. Neural network models will not be available.")
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

# Threshold for averaged probability (CSV batch prediction)
THRESHOLD = 0.85
# Each individual model must also exceed this to reduce false positives
MODEL_AGREE_THRESHOLD = 0.80

# ================= DATABASE =================
def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "fraud_db")
    )

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODELS =================
model_7 = scaler_7 = xgb_7 = None
model_30 = scaler_30 = xgb_30 = None

if HAS_TF:
    try:
        model_7 = load_model(os.path.join(MODEL_DIR, "ann_7_feature.keras"))
        scaler_7 = joblib.load(os.path.join(MODEL_DIR, "scaler_7_feature.joblib"))
        xgb_7 = joblib.load(os.path.join(MODEL_DIR, "xgb_7_feature.joblib"))
        print("[OK] 7 Feature Hybrid Model Loaded")
    except Exception as e:
        print("[WARN] 7 Feature model not loaded:", e)
else:
    print("[SKIP] 7 Feature model (Neural Network) skipped due to missing TensorFlow")

if HAS_TF:
    try:
        model_30 = load_model(os.path.join(MODEL_DIR, "ann_30_feature.keras"))
        scaler_30 = joblib.load(os.path.join(MODEL_DIR, "scaler_30_feature.joblib"))
        xgb_30 = joblib.load(os.path.join(MODEL_DIR, "xgb_30_feature.joblib"))
        print("[OK] 30 Feature Hybrid Model Loaded")
    except Exception as e:
        print("[WARN] 30 Feature model not loaded:", e)
else:
    print("[SKIP] 30 Feature model (Neural Network) skipped due to missing TensorFlow")

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        
        # Stats
        cur.execute("SELECT COUNT(*) as total FROM transactions")
        total_count = cur.fetchone()["total"]
        
        cur.execute("SELECT COUNT(*) as fraud FROM transactions WHERE is_fraud = 1")
        fraud_count = cur.fetchone()["fraud"]
        
        cur.execute("SELECT COUNT(*) as legit FROM transactions WHERE is_fraud = 0")
        legit_count = cur.fetchone()["legit"]
        
        cur.execute("SELECT AVG(risk_score) as avg_risk FROM transactions")
        avg_risk = cur.fetchone()["avg_risk"] or 0
        
        # Recent activity
        cur.execute("SELECT user_id, amount, unix_time, risk_score, status FROM transactions ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        
        for row in rows:
            row["trans_time"] = time.strftime("%H:%M:%S", time.localtime(row["unix_time"]))
            
        cur.close()
        db.close()
        
        return render_template("dashboard.html", 
                               total_count=total_count, 
                               fraud_count=fraud_count, 
                               legit_count=legit_count, 
                               avg_risk=round(avg_risk, 1),
                               recent_data=rows)
    except Exception as e:
        flash(f"Dashboard error: {str(e)}", "danger")
        return redirect(url_for('home'))

# ================= MANUAL PREDICTION =================
@app.route("/manual", methods=["GET", "POST"])
def manual_predict():
    risk = None
    status = None
    prediction = None
    pred_mode = None      # "history" or "model"
    history_count = 0

    if request.method == "POST":
        if model_7 is None or scaler_7 is None or xgb_7 is None:
            flash("7 Feature model not trained yet!", "danger")
            return render_template("manual.html")

        try:
            user_id = request.form["user_id"]
            amount = float(request.form["f1"])
            latitude = float(request.form["f2"])
            longitude = float(request.form["f3"])
            city_pop = float(request.form["f4"])
            unix_time = float(request.form["f5"]) if request.form["f5"] else time.time()
            merch_lat = float(request.form["f6"])
            merch_long = float(request.form["f7"])

            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                flash("Invalid latitude or longitude", "danger")
                return render_template("manual.html")
            if not (-90 <= merch_lat <= 90 and -180 <= merch_long <= 180):
                flash("Invalid merchant latitude or longitude", "danger")
                return render_template("manual.html")
            if amount <= 0:
                flash("Amount must be positive", "danger")
                return render_template("manual.html")

            # ---- Always run models ----
            features = np.array([[amount, latitude, longitude, city_pop, unix_time, merch_lat, merch_long]])
            
            if HAS_TF and model_7 is not None and scaler_7 is not None and xgb_7 is not None:
                scaled = scaler_7.transform(features)
                ann_prob = float(model_7.predict(scaled, verbose=0)[0][0])
                xgb_prob = float(xgb_7.predict_proba(scaled)[0][1])
                model_prob = (ann_prob + xgb_prob) / 2
            elif xgb_7 is not None and scaler_7 is not None:
                # Fallback to XGBoost only
                scaled = scaler_7.transform(features)
                model_prob = float(xgb_7.predict_proba(scaled)[0][1])
                print("[INFO] Using XGBoost-only fallback for manual prediction")
            else:
                flash("Prediction models not available!", "danger")
                return render_template("manual.html")

            # ---- Fetch user history ----
            db = get_db()
            cur = db.cursor()
            cur.execute(
                "SELECT amount, latitude, longitude, risk_score, merch_lat, merch_long, city_pop, unix_time FROM transactions WHERE user_id=%s ORDER BY id DESC",
                (user_id,)
            )
            history = [tuple(float(v) if v is not None else 0.0 for v in row) for row in cur.fetchall()]
            history_count = len(history)

            if history_count > 0:
                # PATH 1: EXISTING USER — full behavioural analysis
                pred_mode = "history"
                amounts    = [r[0] for r in history]
                past_risks = [r[3] for r in history]
                avg_amt    = sum(amounts) / len(amounts)
                avg_risk   = sum(past_risks) / len(past_risks)
                avg_city   = sum(r[6] for r in history) / len(history)

                risk = round(model_prob * 100, 2)

                # --- 1. Amount anomaly ---
                if amount > avg_amt * 3:
                    risk = max(risk, 90)
                elif amount > avg_amt * 2:
                    risk = max(risk, 75)
                elif amount > avg_amt * 1.5:
                    risk = max(risk, 60)

                # --- 2. Customer location anomaly (>1 degree from any past location) ---
                location_anomaly = any(
                    abs(latitude - r[1]) > 1 or abs(longitude - r[2]) > 1
                    for r in history
                )
                if location_anomaly:
                    risk = max(risk, 80)

                # --- 3. Merchant location anomaly (>2 degrees from any past merchant) ---
                merch_anomaly = any(
                    abs(merch_lat - r[4]) > 2 or abs(merch_long - r[5]) > 2
                    for r in history
                )
                if merch_anomaly:
                    risk = max(risk, 75)

                # --- 4. City population mismatch ---
                if avg_city > 0:
                    city_ratio = city_pop / avg_city
                    if city_ratio > 5 or city_ratio < 0.2:
                        risk = max(risk, 65)

                # --- 5. Unusual transaction hour (1am - 5am = high risk hours) ---
                tx_hour = int((unix_time % 86400) / 3600)   # hour of day (UTC)
                if 1 <= tx_hour <= 5:
                    risk = max(risk, risk + 10)             # +10% penalty for odd hours
                    risk = min(risk, 100)

                # --- 6. Blend with historical average risk ---
                risk = round((risk * 0.6 + avg_risk * 0.4), 2)

            else:
                # PATH 2: NEW USER — pure model
                pred_mode = "model"
                risk = round(model_prob * 100, 2)

            # ---- Classify ----
            if risk >= 75:
                status = "HIGH RISK"
                prediction = "FRAUD"
                is_fraud = 1
            elif risk >= 40:
                status = "MEDIUM RISK"
                prediction = "LEGIT"
                is_fraud = 0
            else:
                status = "LOW RISK"
                prediction = "LEGIT"
                is_fraud = 0

            cur.execute("""
                INSERT INTO transactions
                (user_id, amount, latitude, longitude, city_pop, unix_time,
                 merch_lat, merch_long, risk_score, status, is_fraud)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (user_id, amount, latitude, longitude, city_pop, unix_time,
                  merch_lat, merch_long, risk, status, is_fraud))
            db.commit()
            cur.close()
            db.close()

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template("manual.html", risk=risk, status=status,
                           prediction=prediction, pred_mode=pred_mode,
                           history_count=history_count)

# ================= CSV PREDICTION =================
@app.route("/csv", methods=["GET", "POST"])
def csv_predict():
    summary = None
    download_filename = None
    metrics_report = None

    if request.method == "POST":
        if model_30 is None or scaler_30 is None or xgb_30 is None:
            flash("30 Feature model not trained yet!", "danger")
            return render_template("csv.html")

        file = request.files.get("file")
        if not file:
            flash("Please upload a CSV file", "danger")
            return render_template("csv.html")

        try:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df_original = pd.read_csv(path)
            df_original.columns = df_original.columns.str.strip()

            df = df_original.copy()
            if "Class" in df_original.columns:
                y_true = df_original["Class"].values
                df = df.drop("Class", axis=1)
            else:
                y_true = None

            expected_columns = list(scaler_30.feature_names_in_)
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                flash(f"CSV is missing columns: {', '.join(missing_cols)}", "danger")
                return render_template("csv.html")

            df = df[expected_columns]
            X = scaler_30.transform(df.values)

            if HAS_TF and model_30 is not None:
                ann_probs = model_30.predict(X, batch_size=1024, verbose=0).flatten()
                xgb_probs = xgb_30.predict_proba(X)[:,1]
                final_probs = (ann_probs + xgb_probs) / 2

                # Require BOTH models to agree AND averaged prob above threshold
                preds = (
                    (final_probs > THRESHOLD) &
                    (ann_probs > MODEL_AGREE_THRESHOLD) &
                    (xgb_probs > MODEL_AGREE_THRESHOLD)
                ).astype(int)
            else:
                # Fallback to XGBoost only
                xgb_probs = xgb_30.predict_proba(X)[:,1]
                preds = (xgb_probs > THRESHOLD).astype(int)
                print("[INFO] Using XGBoost-only fallback for CSV prediction")

            # Add predictions to original dataframe
            df_original['Prediction'] = ['FRAUD' if p==1 else 'LEGIT' for p in preds]

            # Filter only FRAUD rows for download
            df_fraud = df_original[df_original['Prediction'] == 'FRAUD']

            # Save fraud transactions CSV
            download_filename = "fraud_transactions.csv"
            download_path = os.path.join(UPLOAD_FOLDER, download_filename)
            df_fraud.to_csv(download_path, index=False)

            fraud = int(np.sum(preds == 1))
            legit = int(np.sum(preds == 0))
            summary = {"fraud": fraud, "legit": legit}

            # If Class column exists, show accuracy metrics
            if y_true is not None:
                from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
                acc = accuracy_score(y_true, preds)
                cm = confusion_matrix(y_true, preds)
                cr = classification_report(y_true, preds, digits=4)
                metrics_report = {"accuracy": acc, "confusion_matrix": cm.tolist(), "classification_report": cr}

        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "danger")
            return render_template("csv.html")

    return render_template("csv.html", summary=summary, download_file=download_filename, metrics=metrics_report)

# ================= DOWNLOAD ROUTE =================
@app.route("/download/<filename>")
def download_file_route(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        flash("File not found!", "danger")
        return redirect(url_for('csv_predict'))
    return send_file(path, as_attachment=True)

# ================= HISTORY =================
@app.route("/history")
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT id, user_id, amount, latitude, longitude, city_pop, unix_time, merch_lat, merch_long, risk_score, status
        FROM transactions
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    cur.close()
    db.close()

    columns = ["id","user_id","amount","latitude","longitude","city_pop","unix_time","merchant_lat","merchant_long","risk_score","status"]
    data = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        row_dict["trans_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row_dict["unix_time"]))
        data.append(row_dict)

    return render_template("history.html", data=data)

# ================= DELETE TRANSACTION =================
@app.route("/delete/<int:transaction_id>", methods=["POST"])
def delete_transaction(transaction_id):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("DELETE FROM transactions WHERE id = %s", (transaction_id,))
        db.commit()
        cur.close()
        db.close()
        flash("Transaction deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting transaction: {str(e)}", "danger")
    return redirect(url_for("history"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
