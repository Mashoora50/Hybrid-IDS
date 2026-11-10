 
import matplotlib
matplotlib.use("Agg")
 
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gc
 
FIGURE_DIR = "C:/masters_thesis/figures/hybrid"
os.makedirs(FIGURE_DIR, exist_ok=True)
 
 
print("\n" + "#" * 60)
print("# LOADING MODELS")
print("#" * 60)
 
XGB_DIR = "C:/masters_thesis/artifacts/xgboost_no_leakage"
AE_DIR = "C:/masters_thesis/artifacts/autoencoder_all4days"
 
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{XGB_DIR}/xgb_model.json")
label_encoder = joblib.load(f"{XGB_DIR}/label_encoder.pkl")
xgb_feature_cols = joblib.load(f"{XGB_DIR}/feature_columns.pkl")
xgb_scaler = joblib.load(f"{XGB_DIR}/scaler.pkl")
 
autoencoder = tf.keras.models.load_model(f"{AE_DIR}/autoencoder_model.keras")
ae_scaler = joblib.load(f"{AE_DIR}/scaler.pkl")
ae_feature_cols = joblib.load(f"{AE_DIR}/feature_columns.pkl")
ANOMALY_THRESHOLD = joblib.load(f"{AE_DIR}/anomaly_threshold.pkl")
 
print(f"XGBoost feature count      : {len(xgb_feature_cols)}")
print(f"Autoencoder feature count  : {len(ae_feature_cols)}")
print(f"Attack classes             : {list(label_encoder.classes_)}")
print(f"Anomaly threshold          : {ANOMALY_THRESHOLD:.6f}")
 
 

#  PREPROCESSING

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
 
 
# PIPELINE FUNCTIONS
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
 
 
def run_hybrid_pipeline(df_clean, infil_threshold):
    X_xgb = df_clean[xgb_feature_cols].values
    X_xgb_scaled = xgb_scaler.transform(X_xgb)
    xgb_pred_proba = xgb_model.predict_proba(X_xgb_scaled)
    xgb_pred_encoded = predict_with_infil_threshold(xgb_pred_proba, infil_idx, infil_threshold)
    xgb_pred_labels = label_encoder.inverse_transform(xgb_pred_encoded)
    xgb_confidence = xgb_pred_proba.max(axis=1)
 
    known_attack_mask = xgb_pred_labels != "Benign"
 
    recon_error = np.full(len(df_clean), np.nan)
    benign_idx = np.where(~known_attack_mask)[0]
    if len(benign_idx) > 0:
        X_ae = df_clean.iloc[benign_idx][ae_feature_cols].values
        X_ae_scaled = ae_scaler.transform(X_ae)
 
        CHUNK = 50000
        errors = np.empty(len(X_ae_scaled), dtype=np.float32)
        for start in range(0, len(X_ae_scaled), CHUNK):
            end = start + CHUNK
            chunk_scaled = X_ae_scaled[start:end]
            chunk_pred = autoencoder.predict(chunk_scaled, batch_size=1024, verbose=0)
            errors[start:end] = np.mean(np.square(chunk_scaled - chunk_pred), axis=1)
 
        recon_error[benign_idx] = errors
 
    final_verdict = np.empty(len(df_clean), dtype=object)
    for i in range(len(df_clean)):
        if known_attack_mask[i]:
            final_verdict[i] = f"Known attack: {xgb_pred_labels[i]}"
        elif recon_error[i] > ANOMALY_THRESHOLD:
            final_verdict[i] = "Suspicious -- possible unknown/zero-day attack"
        else:
            final_verdict[i] = "Benign"
 
    out = pd.DataFrame({
        "xgb_prediction": xgb_pred_labels,
        "xgb_confidence": xgb_confidence.round(4),
        "reconstruction_error": recon_error,
        "final_verdict": final_verdict,
    }, index=df_clean.index)
    return out
 
 
def predicted_category(verdict):
    if verdict.startswith("Known attack"):
        return "flagged_known_attack"
    elif verdict.startswith("Suspicious"):
        return "flagged_unknown_attack"
    else:
        return "flagged_benign"
 
 
KNOWN_ATTACK_LABELS = ["Bot", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "Infilteration"]
 
 
def true_category(label):
    if label == "Benign":
        return "actual_benign"
    elif label in KNOWN_ATTACK_LABELS:
        return "actual_known_attack"
    else:
        return "actual_unknown_attack"
 
 

print("\n" + "#" * 60)
print("# LOADING DATA")
print("#" * 60)
 
raw_df1 = pd.read_csv("C:/masters_thesis/Data/raw/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", low_memory=False)
raw_df2 = pd.read_csv("C:/masters_thesis/Data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", low_memory=False)
raw_df = pd.concat([raw_df1, raw_df2], ignore_index=True)
del raw_df1, raw_df2
gc.collect()
print(f"Loaded and combined raw rows: {len(raw_df):,}")
 
true_labels_full = raw_df["Label"].reset_index(drop=True)
 
print("Cleaning data (this happens once)...")
df_clean = preprocess_raw_flows(raw_df)
del raw_df
gc.collect()
true_labels = true_labels_full.loc[df_clean.index]
print(f"Cleaned dataset: {len(df_clean):,} rows")
 

# SECTION: HYBRID EVALUATION
print("\n" + "#" * 60)
print("# HYBRID EVALUATION")
print("#" * 60)
 
pred = run_hybrid_pipeline(df_clean, infil_threshold=0.9)
hybrid_results = df_clean.copy()
hybrid_results["xgb_prediction"] = pred["xgb_prediction"].values
hybrid_results["xgb_confidence"] = pred["xgb_confidence"].values
hybrid_results["reconstruction_error"] = pred["reconstruction_error"].values
hybrid_results["final_verdict"] = pred["final_verdict"].values
hybrid_results["true_label"] = true_labels.values
 
print("\nVerdict counts (threshold=0.9):")
print(hybrid_results["final_verdict"].value_counts())
 
hybrid_results["predicted_category"] = hybrid_results["final_verdict"].apply(predicted_category)
hybrid_results["true_category"] = hybrid_results["true_label"].apply(true_category)
 
crosstab = pd.crosstab(hybrid_results["true_category"], hybrid_results["predicted_category"])
 
print("\nConfusion table (threshold=0.9):")
print(crosstab)
 
for cat in ["actual_benign", "actual_known_attack", "actual_unknown_attack"]:
    if cat in crosstab.index:
        row = crosstab.loc[cat]
        total = row.sum()
        print(f"\n{cat}  (total flows = {total}):")
        for col, count in row.items():
            print(f"   -> {col:<25s}: {count:>8,}  ({count/total:.2%})")
 
#  hybrid pipeline confusion matrix
cat_order_true = ["actual_benign", "actual_known_attack", "actual_unknown_attack"]
cat_order_pred = ["flagged_benign", "flagged_known_attack", "flagged_unknown_attack"]
cm_display = crosstab.reindex(index=cat_order_true, columns=cat_order_pred, fill_value=0)
 
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm_display.values, cmap="Blues")
ax.set_xticks(range(len(cat_order_pred)))
ax.set_yticks(range(len(cat_order_true)))
ax.set_xticklabels(cat_order_pred, rotation=30, ha="right")
ax.set_yticklabels(cat_order_true)
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category")
ax.set_title("Hybrid Pipeline Confusion Matrix (Infil. threshold = 0.9)")
 
for i in range(cm_display.shape[0]):
    for j in range(cm_display.shape[1]):
        val = cm_display.values[i, j]
        color = "white" if val > cm_display.values.max() / 2 else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center", color=color)
 
plt.colorbar(im, ax=ax, label="Flow count")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/hybrid_confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"\nSaved figure: {FIGURE_DIR}/hybrid_confusion_matrix.png")
 
del hybrid_results, pred
gc.collect()
 
 

#  THRESHOLD SWEEP

print("\n" + "#" * 60)
print("# THRESHOLD SWEEP")
print("#" * 60)
 
sweep_results = []
 
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = run_hybrid_pipeline(df_clean, infil_threshold=thresh)
 
    pred_cat = pred["final_verdict"].apply(predicted_category)
    true_cat = true_labels.apply(true_category).values
 
    ct = pd.crosstab(pd.Series(true_cat, name="true_category"), pred_cat.values)
 
    known_row = ct.loc["actual_known_attack"]
    known_total = known_row.sum()
    known_caught = known_row.get("flagged_known_attack", 0) + known_row.get("flagged_unknown_attack", 0)
    known_recall = known_caught / known_total
 
    unseen_row = ct.loc["actual_unknown_attack"]
    unseen_total = unseen_row.sum()
    unseen_caught = unseen_row.get("flagged_unknown_attack", 0) + unseen_row.get("flagged_known_attack", 0)
    unseen_recall = unseen_caught / unseen_total
 
    benign_row = ct.loc["actual_benign"]
    benign_total = benign_row.sum()
    benign_false_alarm = 1 - (benign_row.get("flagged_benign", 0) / benign_total)
 
    sweep_results.append({
        "threshold": thresh,
        "known_attack_recall": round(known_recall, 4),
        "unseen_attack_recall": round(unseen_recall, 4),
        "benign_false_alarm_rate": round(benign_false_alarm, 4),
    })
 
    print(f"threshold={thresh}  |  known-attack recall={known_recall:.2%}  |  "
          f"unseen-attack recall={unseen_recall:.2%}  |  benign false-alarm={benign_false_alarm:.2%}")
 
    del pred, pred_cat, ct
    gc.collect()
 
sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(f"{FIGURE_DIR}/hybrid_threshold_sweep.csv", index=False)
print(f"\nSaved: {FIGURE_DIR}/hybrid_threshold_sweep.csv")
 
# threshold sweep chart 
plt.figure(figsize=(9, 6))
plt.plot(sweep_df["threshold"], sweep_df["known_attack_recall"], marker="o", label="Known-attack recall")
plt.plot(sweep_df["threshold"], sweep_df["unseen_attack_recall"], marker="o", label="Unseen-attack recall")
plt.plot(sweep_df["threshold"], sweep_df["benign_false_alarm_rate"], marker="o", label="Benign false-alarm rate")
plt.xlabel("Infiltration confidence threshold")
plt.ylabel("Rate")
plt.title("Hybrid Pipeline -- Infiltration Confidence Threshold Sweep")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/threshold_sweep.png", dpi=150)
plt.close()
print(f"Saved: {FIGURE_DIR}/threshold_sweep.png")
 
 
print("\n" + "#" * 60)
print("# SAVE CONFIG")
print("#" * 60)
 
CHOSEN_THRESHOLD = 0.6
 
HYBRID_DIR = "C:/masters_thesis/artifacts/hybrid"
os.makedirs(HYBRID_DIR, exist_ok=True)
 
hybrid_config = {
    "infil_confidence_threshold": CHOSEN_THRESHOLD,
    "autoencoder_artifact_dir": AE_DIR,
    "xgb_artifact_dir": XGB_DIR,
    "notes": "See hybrid_threshold_sweep.csv for the full sweep this run produced.",
}
 
joblib.dump(hybrid_config, f"{HYBRID_DIR}/hybrid_config.pkl")
 
print(f"Saved config to: {HYBRID_DIR}/hybrid_config.pkl")
for k, v in hybrid_config.items():
    print(f"  {k}: {v}")
 
 

print("\n" + "#" * 60)
print("# ARTIFACT SUMMARY")
print("#" * 60)
 
 
def folder_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)
 
 
ARTIFACTS_ROOT = "C:/masters_thesis/artifacts"
for name in sorted(os.listdir(ARTIFACTS_ROOT)):
    full_path = os.path.join(ARTIFACTS_ROOT, name)
    if os.path.isdir(full_path):
        print(f"  {name:<25s} ({folder_size_mb(full_path):.1f} MB)")
 
print("\nDONE.")
 
