# =====================================================
# FRAUD DETECTION TRAINING - HIGH THRESHOLD & TP/FP
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("model_files", exist_ok=True)

def train_pipeline(df, feature_columns, model_tag):
    print(f"\n========== Training {model_tag} ==========")

    X = df[feature_columns]
    y = df["Class"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE:")
    print(pd.Series(y_train_res).value_counts())

    # ================= ANN =================
    input_dim = X_train_res.shape[1]
    ann = Sequential([
        Input(shape=(input_dim,)),
        Dense(128 if input_dim>10 else 32, activation='relu'),
        Dense(64 if input_dim>10 else 16, activation='relu'),
        Dense(32 if input_dim>10 else 8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ann.fit(X_train_res, y_train_res, epochs=25, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=1)

    # ================= XGBOOST =================
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss", random_state=42)
    xgb.fit(X_train_res, y_train_res)

    # ================= THRESHOLD & COMBINED PREDICTION =================
    ann_probs = ann.predict(X_test).flatten()
    xgb_probs = xgb.predict_proba(X_test)[:,1]
    combined_prob = (ann_probs + xgb_probs)/2

    # --- Try multiple thresholds to analyze ---
    print("\n🔹 Threshold Analysis (TP/FP vs Threshold)")
    for TH in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        preds = (combined_prob > TH).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = accuracy_score(y_test, preds)
        print(f"Threshold {TH:.2f} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} | Accuracy: {acc:.4f}")

    # --- Save final predictions with default threshold 0.5 ---
    TH_DEFAULT = 0.5
    final_preds_default = (combined_prob > TH_DEFAULT).astype(int)
    df_default = pd.DataFrame(X_test, columns=feature_columns)
    df_default["Class"] = y_test.values
    df_default["Prediction"] = np.where(final_preds_default==1, "FRAUD", "LEGIT")
    df_default.to_csv(f"model_files/final_{model_tag}_predictions.csv", index=False)
    print(f"\n✅ Default threshold predictions saved to model_files/final_{model_tag}_predictions.csv")

    # --- Save high-threshold predictions to reduce false positives ---
    HIGH_THRESHOLD = 0.75  # Change this value to reduce FP further
    final_preds_high = (combined_prob > HIGH_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_preds_high).ravel()
    print(f"\nHigh Threshold {HIGH_THRESHOLD} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print("Classification Report:\n", classification_report(y_test, final_preds_high))

    df_high = pd.DataFrame(X_test, columns=feature_columns)
    df_high["Class"] = y_test.values
    df_high["Prediction"] = np.where(final_preds_high==1, "FRAUD", "LEGIT")
    df_high.to_csv(f"model_files/final_{model_tag}_predictions_high_threshold.csv", index=False)
    print(f"✅ High threshold predictions saved to model_files/final_{model_tag}_predictions_high_threshold.csv")

    # Save models
    ann.save(f"model_files/ann_{model_tag}.keras")
    joblib.dump(xgb, f"model_files/xgb_{model_tag}.joblib")
    joblib.dump(scaler, f"model_files/scaler_{model_tag}.joblib")
    print(f"✅ {model_tag} Models Saved Successfully!\n")

# =====================================================
# Train 30 feature model
# =====================================================
df_30 = pd.read_csv("creditcard.csv")
features_30 = df_30.drop("Class", axis=1).columns
train_pipeline(df_30, features_30, "30_feature")

# =====================================================
# Train 7 feature model
# =====================================================
df_7 = pd.read_csv("manual_dataset.csv")
features_7 = ['Amount','Latitude','Longitude','City_Pop','Unix_Time','Merch_Lat','Merch_Long']
train_pipeline(df_7, features_7, "7_feature")

print("\n🔥 ALL MODELS TRAINED SUCCESSFULLY WITH HIGH THRESHOLD PREDICTIONS 🔥")