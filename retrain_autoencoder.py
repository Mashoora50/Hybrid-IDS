import matplotlib
matplotlib.use("Agg")  
 
import pandas as pd
import numpy as np
import gc
 
DATA_PATHS = [
    "C:/masters_thesis/Data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "C:/masters_thesis/Data/raw/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "C:/masters_thesis/Data/raw/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "C:/masters_thesis/Data/raw/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]
 
REDUCED_FEATURE_COLUMNS = [
    'dst_port', 'protocol', 'tot_fwd_pkts', 'totlen_fwd_pkts', 'fwd_pkt_len_min',
    'bwd_pkt_len_min', 'flow_byts/s', 'flow_pkts/s', 'fwd_iat_tot', 'fwd_iat_mean',
    'fwd_iat_std', 'fwd_iat_max', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std',
    'bwd_iat_min', 'fwd_psh_flags', 'bwd_pkts/s', 'pkt_len_max', 'fin_flag_cnt',
    'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count',
    'down/up_ratio', 'fwd_seg_size_avg', 'subflow_bwd_byts', 'init_fwd_win_byts',
    'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean',
    'active_std', 'idle_std', 'hour', 'day_of_week',
]
 

FIGURE_DIR = "C:/masters_thesis/figures/autoencoder"
import os
os.makedirs(FIGURE_DIR, exist_ok=True)
 
cleaned_pieces = []
 
for path in DATA_PATHS:
    fname = path.split("/")[-1]
    print(f"Processing: {fname}")
 
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df["label"] != "Label"]
 
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = df.drop(columns=["timestamp"])
 
    keep_cols = [c for c in REDUCED_FEATURE_COLUMNS if c in df.columns] + ["label"]
    df = df[keep_cols]  # shrink down immediately -- drop the other ~45 unused columns
 
    feature_cols_present = [c for c in REDUCED_FEATURE_COLUMNS if c in df.columns]
    df[feature_cols_present] = df[feature_cols_present].apply(pd.to_numeric, errors="coerce")
    df[feature_cols_present] = df[feature_cols_present].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols_present)
    df = df.drop_duplicates().reset_index(drop=True)
 
    print(f"  -> {df.shape[0]:,} rows kept, {df.shape[1]} columns")
    cleaned_pieces.append(df)
 
    del df  # free the memory before loading the next file
    gc.collect()
 
print("\nCombining all cleaned pieces...")
df = pd.concat(cleaned_pieces, ignore_index=True)
del cleaned_pieces
gc.collect()
 
print("Combined shape (all 4 days, reduced):", df.shape)
print(df["label"].value_counts())
 
feature_cols = [c for c in REDUCED_FEATURE_COLUMNS if c in df.columns]
constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
if constant_cols:
    print("Dropping constant columns:", constant_cols)
    df = df.drop(columns=constant_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]
else:
    print("No constant columns -- day_of_week now carries real signal across the 4 days.")
 
print("Final feature count:", len(feature_cols))
print(feature_cols)
 
BENIGN_LABEL = "Benign"
TRUE_UNSEEN_LABELS = ["DoS attacks-GoldenEye", "DoS attacks-Slowloris"]
 
benign_df = df[df["label"] == BENIGN_LABEL].copy()
unseen_df = df[df["label"].isin(TRUE_UNSEEN_LABELS)].copy()
infil_df = df[df["label"] == "Infilteration"].copy()
 
print("Benign rows (all 4 days):", len(benign_df))
print("True unseen rows:", len(unseen_df), dict(unseen_df["label"].value_counts()))
print("Infiltration rows (diagnostic):", len(infil_df))
 

#  Train/val/test split

from sklearn.model_selection import train_test_split
 
RANDOM_STATE = 42
 
X_benign = benign_df[feature_cols].values
X_train, X_temp = train_test_split(X_benign, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test_benign = train_test_split(X_temp, test_size=0.50, random_state=RANDOM_STATE)
 
X_test_unseen = unseen_df[feature_cols].values
X_infil = infil_df[feature_cols].values
 
print("Train (benign):", X_train.shape)
print("Val   (benign):", X_val.shape)
print("Test  (benign):", X_test_benign.shape)
print("Test  (true unseen attack):", X_test_unseen.shape)
 

#  Scaling

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
scaler.fit(X_train)
 
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_benign_s = scaler.transform(X_test_benign)
X_test_unseen_s = scaler.transform(X_test_unseen)
X_infil_s = scaler.transform(X_infil)
 
X_test_s = np.vstack([X_test_benign_s, X_test_unseen_s])
y_test = np.concatenate([np.zeros(len(X_test_benign_s)), np.ones(len(X_test_unseen_s))])
 

#  Build & training

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
 
tf.random.set_seed(RANDOM_STATE)
 
def build_autoencoder(input_dim, encoding_dim=16, l2_reg=1e-5):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu", activity_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.Dense(32, activation="relu")(x)
    bottleneck = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
 
input_dim = X_train_s.shape[1]
autoencoder = build_autoencoder(input_dim, encoding_dim=16)
autoencoder.summary()
 
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
 
print("\nStarting training...")
history = autoencoder.fit(
    X_train_s, X_train_s,
    validation_data=(X_val_s, X_val_s),
    epochs=100, batch_size=512, shuffle=True,
    callbacks=[early_stop, reduce_lr], verbose=1,
)
 

# Training vs validation loss curve

import matplotlib.pyplot as plt
 
plt.figure(figsize=(9, 5.5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Autoencoder Training vs Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/training_validation_loss.png", dpi=150)
plt.show()
 

#  Threshold from validation reconstruction error

def reconstruction_error(model, X):
    X_pred = model.predict(X, batch_size=1024, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)
 
val_errors = reconstruction_error(autoencoder, X_val_s)
THRESHOLD = np.percentile(val_errors, 95)
print(f"\nSelected THRESHOLD (95th pct): {THRESHOLD:.6f}")
 

plt.figure(figsize=(9, 5.5))
plt.hist(val_errors, bins=150, color="#4C72B0", alpha=0.85)
plt.axvline(THRESHOLD, color="black", linestyle="--", linewidth=2,
            label=f"Threshold (95th pct) = {THRESHOLD:.4f}")
plt.xscale("log")
plt.title("Benign Validation Reconstruction Error Distribution")
plt.xlabel("Reconstruction MSE (log scale)")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/benign_validation_reconstruction_error.png", dpi=150)
plt.show()
 
#evaluation: benign vs. true unseen attacks

from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
 
test_errors = reconstruction_error(autoencoder, X_test_s)
y_pred = (test_errors > THRESHOLD).astype(int)
 
print("\n=== PRIMARY: Benign (ALL 4 DAYS) vs. True Unseen Attacks ===")
print(classification_report(y_test, y_pred, target_names=["Benign", "Unseen Attack"]))
roc_auc = roc_auc_score(y_test, test_errors)
print("ROC-AUC:", roc_auc)
 
# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Unseen Attack"]).plot(
    cmap="Blues", values_format="d", ax=ax)
plt.title("Confusion Matrix — Autoencoder Trained on All 4 Days")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/confusion_matrix_all4days.png", dpi=150)
plt.show()
 
# Reconstruction error distribution, benign vs. unseen attack (test set) 
# This is the log-scale overlaid histogram — benign vs. attack, threshold marked.
benign_test_errors = reconstruction_error(autoencoder, X_test_benign_s)
unseen_test_errors = reconstruction_error(autoencoder, X_test_unseen_s)
 
plt.figure(figsize=(10, 6))
bins = np.logspace(np.log10(max(test_errors.min(), 1e-6)), np.log10(test_errors.max()), 150)
plt.hist(benign_test_errors, bins=bins, density=True, alpha=0.6, color="#4C72B0", label="Benign (test)")
plt.hist(unseen_test_errors, bins=bins, density=True, alpha=0.6, color="#C44E52", label="Unseen Attack (test)")
plt.axvline(THRESHOLD, color="black", linestyle="--", linewidth=2, label=f"Threshold = {THRESHOLD:.4f}")
plt.xscale("log")
plt.title("Reconstruction Error Distribution — Benign vs. True Unseen Attacks")
plt.xlabel("Reconstruction MSE (log scale)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/reconstruction_error_distribution.png", dpi=150)
plt.show()
 
# ROC curve 
fpr, tpr, roc_thresholds = roc_curve(y_test, test_errors)
 
plt.figure(figsize=(7, 6.5))
plt.plot(fpr, tpr, color="#C44E52", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
# mark the operating point corresponding to the chosen 95th-pct threshold
op_idx = np.argmin(np.abs(roc_thresholds - THRESHOLD))
plt.scatter(fpr[op_idx], tpr[op_idx], color="black", zorder=5,
            label=f"Operating point (95th pct threshold)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Autoencoder Anomaly Detection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/roc_curve.png", dpi=150)
plt.show()
 
precision, recall, pr_thresholds = precision_recall_curve(y_test, test_errors)
avg_precision = average_precision_score(y_test, test_errors)
 
plt.figure(figsize=(7, 6.5))
plt.plot(recall, precision, color="#55A868", linewidth=2,
         label=f"PR curve (AP = {avg_precision:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Autoencoder Anomaly Detection")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/precision_recall_curve.png", dpi=150)
plt.show()
 

#  DIAGNOSTIC: Infiltration

infil_errors = reconstruction_error(autoencoder, X_infil_s)
infil_catch_rate = (infil_errors > THRESHOLD).mean()
print(f"\nInfiltration flagged by autoencoder alone: {infil_catch_rate:.2%}")
 
# ---- PLOT: Infiltration reconstruction error vs. benign/threshold (diagnostic only) ----
plt.figure(figsize=(10, 6))
plt.hist(benign_test_errors, bins=bins, density=True, alpha=0.5, color="#4C72B0", label="Benign (test)")
plt.hist(infil_errors, bins=bins, density=True, alpha=0.5, color="#DD8452", label="Infiltration (diagnostic)")
plt.axvline(THRESHOLD, color="black", linestyle="--", linewidth=2, label=f"Threshold = {THRESHOLD:.4f}")
plt.xscale("log")
plt.title(f"Reconstruction Error — Infiltration Diagnostic (caught: {infil_catch_rate:.1%})")
plt.xlabel("Reconstruction MSE (log scale)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/infiltration_diagnostic_distribution.png", dpi=150)
plt.show()
 

import joblib
 
AE_DIR = "C:/masters_thesis/artifacts/autoencoder_all4days"
os.makedirs(AE_DIR, exist_ok=True)
 
autoencoder.save(f"{AE_DIR}/autoencoder_model.keras")
joblib.dump(scaler, f"{AE_DIR}/scaler.pkl")
joblib.dump(feature_cols, f"{AE_DIR}/feature_columns.pkl")
joblib.dump(THRESHOLD, f"{AE_DIR}/anomaly_threshold.pkl")
 
print("\nSaved model artifacts to:", AE_DIR)
print("Files:", os.listdir(AE_DIR))
print("\nSaved figures to:", FIGURE_DIR)
print("Files:", os.listdir(FIGURE_DIR))
 































 