import matplotlib
matplotlib.use("Agg")
 
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import os
import gc
 
FIGURE_DIR = "C:/masters_thesis/figures/shap"
os.makedirs(FIGURE_DIR, exist_ok=True)
 
XGB_DIR = "C:/masters_thesis/artifacts/xgboost_no_leakage"
AE_DIR = "C:/masters_thesis/artifacts/autoencoder_all4days"
HYBRID_DIR = "C:/masters_thesis/artifacts/hybrid"
 
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{XGB_DIR}/xgb_model.json")
label_encoder = joblib.load(f"{XGB_DIR}/label_encoder.pkl")
xgb_feature_cols = joblib.load(f"{XGB_DIR}/feature_columns.pkl")
xgb_scaler = joblib.load(f"{XGB_DIR}/scaler.pkl")
 
autoencoder = tf.keras.models.load_model(f"{AE_DIR}/autoencoder_model.keras")
ae_scaler = joblib.load(f"{AE_DIR}/scaler.pkl")
ae_feature_cols = joblib.load(f"{AE_DIR}/feature_columns.pkl")
ANOMALY_THRESHOLD = joblib.load(f"{AE_DIR}/anomaly_threshold.pkl")
 
hybrid_config = joblib.load(f"{HYBRID_DIR}/hybrid_config.pkl")
INFIL_CONFIDENCE_THRESHOLD = hybrid_config["infil_confidence_threshold"]
 
print("Loaded XGBoost (no-leakage), Autoencoder, and hybrid config.")
print("Infiltration confidence threshold:", INFIL_CONFIDENCE_THRESHOLD)
print("XGBoost classes:", list(label_encoder.classes_))
print("XGBoost features (should NOT include day_of_week/hour):", xgb_feature_cols)
 
 
def preprocess_raw_flows(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    df = df[df["label"] != "Label"]
 
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df = df.drop(columns=["timestamp"])
 
    required_cols = sorted(set(xgb_feature_cols) | set(ae_feature_cols))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
 
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce", downcast="float")
    df[required_cols] = df[required_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    return df
 
 
infil_idx = list(label_encoder.classes_).index("Infilteration")
 
 
def predict_with_infil_threshold(y_proba, infil_idx, infil_min_conf):
    y_pred = np.argmax(y_proba, axis=1)
    infil_picked = (y_pred == infil_idx)
    low_conf = infil_picked & (y_proba[:, infil_idx] < infil_min_conf)
    if low_conf.any():
        proba_without_infil = y_proba.copy()
        proba_without_infil[:, infil_idx] = -1
        y_pred[low_conf] = np.argmax(proba_without_infil[low_conf], axis=1)
    return y_pred
 
 
def hybrid_predict(df_raw):
    df = preprocess_raw_flows(df_raw)
 
    X_xgb = df[xgb_feature_cols].values
    X_xgb_scaled = xgb_scaler.transform(X_xgb)
    xgb_pred_proba = xgb_model.predict_proba(X_xgb_scaled)
    xgb_pred_encoded = predict_with_infil_threshold(xgb_pred_proba, infil_idx, INFIL_CONFIDENCE_THRESHOLD)
    xgb_pred_labels = label_encoder.inverse_transform(xgb_pred_encoded)
    xgb_confidence = xgb_pred_proba.max(axis=1)
 
    known_attack_mask = xgb_pred_labels != "Benign"
 
    recon_error = np.full(len(df), np.nan)
    benign_idx = np.where(~known_attack_mask)[0]
    if len(benign_idx) > 0:
        X_ae = df.iloc[benign_idx][ae_feature_cols].values
        X_ae_scaled = ae_scaler.transform(X_ae)
 
        CHUNK = 50000
        errors = np.empty(len(X_ae_scaled), dtype=np.float32)
        for start in range(0, len(X_ae_scaled), CHUNK):
            end = start + CHUNK
            chunk_scaled = X_ae_scaled[start:end]
            chunk_pred = autoencoder.predict(chunk_scaled, batch_size=1024, verbose=0)
            errors[start:end] = np.mean(np.square(chunk_scaled - chunk_pred), axis=1)
        recon_error[benign_idx] = errors
 
    final_verdict = []
    for i in range(len(df)):
        if known_attack_mask[i]:
            final_verdict.append(f"Known attack: {xgb_pred_labels[i]}")
        elif recon_error[i] > ANOMALY_THRESHOLD:
            final_verdict.append("Suspicious -- possible unknown/zero-day attack")
        else:
            final_verdict.append("Benign")
 
    results = df
    results["xgb_prediction"] = xgb_pred_labels
    results["xgb_confidence"] = xgb_confidence.round(4)
    results["reconstruction_error"] = recon_error
    results["final_verdict"] = final_verdict
    return results, X_xgb_scaled
 
 
raw_df1 = pd.read_csv("C:/masters_thesis/Data/raw/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", low_memory=False)
raw_df2 = pd.read_csv("C:/masters_thesis/Data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", low_memory=False)
raw_df = pd.concat([raw_df1, raw_df2], ignore_index=True)
del raw_df1, raw_df2
gc.collect()
 
hybrid_results, X_xgb_scaled_all = hybrid_predict(raw_df)
del raw_df
gc.collect()
print(hybrid_results["final_verdict"].value_counts())
 
 
# ---- shap on xgboost ----
explainer = shap.TreeExplainer(xgb_model)
 
SAMPLE_SIZE = 5000
sample_idx = np.random.RandomState(42).choice(len(X_xgb_scaled_all), size=min(SAMPLE_SIZE, len(X_xgb_scaled_all)), replace=False)
X_sample = X_xgb_scaled_all[sample_idx]
 
shap_values_sample = explainer.shap_values(X_sample)
print("SHAP values shape/type:", type(shap_values_sample),
      np.array(shap_values_sample).shape if not isinstance(shap_values_sample, list) else len(shap_values_sample))
 
if isinstance(shap_values_sample, list):
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_sample], axis=0)
elif shap_values_sample.ndim == 3:
    mean_abs_shap = np.abs(shap_values_sample).mean(axis=(0, 2))
else:
    mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
 
importance_df = pd.DataFrame({
    "feature": xgb_feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)
 
print(importance_df.head(15))
 
plt.figure(figsize=(9, 6))
plt.barh(importance_df["feature"].head(15)[::-1], importance_df["mean_abs_shap"].head(15)[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Global Feature Importance -- XGBoost, No Leakage (SHAP)")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/shap_global_importance.png", dpi=150)
plt.close()
print(f"Saved: {FIGURE_DIR}/shap_global_importance.png")
 
 
def get_class_shap_row(sv, pred_class_idx):
    if isinstance(sv, list):
        return sv[pred_class_idx][0]
    elif sv.ndim == 3:
        return sv[0, :, pred_class_idx]
    else:
        return sv[0]
 
 
def explain_prediction(row_index, top_n=5):
    x_row = X_xgb_scaled_all[row_index:row_index + 1]
    row = hybrid_results.iloc[row_index]
    pred_label = row["xgb_prediction"]
    pred_class_idx = list(label_encoder.classes_).index(pred_label)
 
    sv = explainer.shap_values(x_row)
    row_shap = get_class_shap_row(sv, pred_class_idx)
 
    explanation = pd.DataFrame({
        "feature": xgb_feature_cols,
        "value": x_row[0],
        "shap_contribution": row_shap
    }).sort_values("shap_contribution", key=abs, ascending=False).head(top_n)
 
    print(f"Flow #{row_index}")
    print(f"  Final verdict:     {row['final_verdict']}")
    print(f"  XGBoost predicted: {pred_label} (confidence: {row['xgb_confidence']:.2%})")
    if not np.isnan(row['reconstruction_error']):
        print(f"  Reconstruction err: {row['reconstruction_error']:.4f}")
    print(f"  Top {top_n} contributing features (toward '{pred_label}'):")
    print(explanation.to_string(index=False))
    return explanation
 
 
known_attack_positions = np.where(hybrid_results["final_verdict"].str.startswith("Known attack"))[0][:3]
print("=" * 60)
print("KNOWN ATTACK examples (caught directly by XGBoost)")
print("=" * 60)
for pos in known_attack_positions:
    explain_prediction(pos)
    print()
 
suspicious_positions = np.where(hybrid_results["final_verdict"].str.startswith("Suspicious"))[0][:3]
print("=" * 60)
print("SUSPICIOUS examples (XGBoost said Benign, autoencoder disagreed)")
print("=" * 60)
for pos in suspicious_positions:
    explain_prediction(pos)
    print()
 
if len(known_attack_positions) > 0:
    example_idx = known_attack_positions[0]
    x_example = X_xgb_scaled_all[example_idx:example_idx + 1]
    pred_label_example = hybrid_results.iloc[example_idx]["xgb_prediction"]
    pred_class_idx_example = list(label_encoder.classes_).index(pred_label_example)
 
    sv_example = explainer.shap_values(x_example)
    row_shap_example = get_class_shap_row(sv_example, pred_class_idx_example)
 
    ev = explainer.expected_value
    expected_value = ev[pred_class_idx_example] if isinstance(ev, (list, np.ndarray)) else ev
 
    print(f"\nSaving waterfall plot for Flow #{example_idx} -- predicted '{pred_label_example}'")
    shap.plots.waterfall(
        shap.Explanation(
            values=row_shap_example,
            base_values=expected_value,
            data=x_example[0],
            feature_names=xgb_feature_cols
        ),
        max_display=10,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/shap_waterfall_example.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURE_DIR}/shap_waterfall_example.png")
 
 
# ---- shap on auto encoder ----
def explain_anomaly(row_index, top_n=5):
    row = hybrid_results.iloc[row_index]
 
    if not row["final_verdict"].startswith("Suspicious"):
        print(f"Flow #{row_index} was not flagged by the autoencoder "
              f"(verdict: {row['final_verdict']}) -- nothing to explain here.")
        return None
 
    x_raw = row[ae_feature_cols].values.astype(float).reshape(1, -1)
    x_scaled = ae_scaler.transform(x_raw)
    x_reconstructed = autoencoder.predict(x_scaled, verbose=0)
 
    per_feature_error = np.square(x_scaled[0] - x_reconstructed[0])
 
    breakdown = pd.DataFrame({
        "feature": ae_feature_cols,
        "actual_value": x_raw[0],
        "scaled_value": x_scaled[0],
        "reconstructed_value": x_reconstructed[0],
        "squared_error": per_feature_error
    }).sort_values("squared_error", ascending=False).head(top_n)
 
    print(f"Flow #{row_index}")
    print(f"  Final verdict:      {row['final_verdict']}")
    print(f"  Total recon. error: {row['reconstruction_error']:.4f} "
          f"(threshold: {ANOMALY_THRESHOLD:.4f})")
    print(f"  Top {top_n} features driving the anomaly:")
    print(breakdown.to_string(index=False))
    return breakdown
 
 
suspicious_positions = np.where(hybrid_results["final_verdict"].str.startswith("Suspicious"))[0][:3]
for pos in suspicious_positions:
    explain_anomaly(pos)
    print()
 
 
ATTACK_SEVERITY_WEIGHT = {
    "Infilteration": 0.95,
    "DoS attacks-Hulk": 0.85,
    "Bot": 0.75,
    "DoS attacks-SlowHTTPTest": 0.65,
}
 
 
def compute_risk_score(row):
    verdict = row["final_verdict"]
    confidence = row["xgb_confidence"]
    recon_error = row["reconstruction_error"]
 
    if verdict.startswith("Known attack"):
        attack_type = row["xgb_prediction"]
        severity_weight = ATTACK_SEVERITY_WEIGHT.get(attack_type, 0.7)
        score = 50 + 50 * confidence * severity_weight
    elif verdict.startswith("Suspicious"):
        ratio = recon_error / ANOMALY_THRESHOLD if ANOMALY_THRESHOLD > 0 else 1
        score = min(90, 30 + 30 * min(ratio, 2))
    else:  # Benign
        ratio = recon_error / ANOMALY_THRESHOLD if not np.isnan(recon_error) and ANOMALY_THRESHOLD > 0 else 0
        score = min(15, 15 * ratio)
 
    return round(score, 1)
 
 
def risk_level(score):
    if score >= 76:
        return "Critical"
    elif score >= 51:
        return "High"
    elif score >= 26:
        return "Medium"
    else:
        return "Low"
 
 
hybrid_results["risk_score"] = hybrid_results.apply(compute_risk_score, axis=1)
hybrid_results["risk_level"] = hybrid_results["risk_score"].apply(risk_level)
 
print(hybrid_results["risk_level"].value_counts())
print()
print(hybrid_results[["final_verdict", "xgb_prediction", "xgb_confidence",
                       "reconstruction_error", "risk_score", "risk_level"]].head(20))
 
 
# ---- combining shap and risk score ----
def full_alert_report(row_index):
    row = hybrid_results.iloc[row_index]
    print(f"=== ALERT -- Flow #{row_index} ===")
    print(f"Verdict:      {row['final_verdict']}")
    print(f"Risk Score:   {row['risk_score']} / 100")
    print(f"Risk Level:   {row['risk_level']}")
    print(f"Confidence:   {row['xgb_confidence']:.2%}")
    print()
    explain_prediction(row_index, top_n=5)
 
 
top_risk_idx = hybrid_results["risk_score"].idxmax()
full_alert_report(hybrid_results.index.get_loc(top_risk_idx))
 
 
# ---- save results ----
RESULTS_DIR = "C:/masters_thesis/notebook"
os.makedirs(RESULTS_DIR, exist_ok=True)
 
output_cols = ["xgb_prediction", "xgb_confidence", "reconstruction_error",
               "final_verdict", "risk_score", "risk_level"]
hybrid_results[output_cols].to_csv(f"{RESULTS_DIR}/hybrid_risk_scored_results.csv", index=False)
print("Saved to:", f"{RESULTS_DIR}/hybrid_risk_scored_results.csv")
 
print("\nDONE.")
 