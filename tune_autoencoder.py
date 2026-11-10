import matplotlib
matplotlib.use("Agg")
 
import pandas as pd
import numpy as np
import gc
import os
import time
 
FIGURE_DIR = "C:/masters_thesis/figures/autoencoder_tuning"
os.makedirs(FIGURE_DIR, exist_ok=True)
 
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
 

#  LOADING AND CLEANING DATA 
print("\n" + "#" * 60)
print("# LOADING AND CLEANING DATA")
print("#" * 60)
 
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
    df = df[keep_cols]
 
    feature_cols_present = [c for c in REDUCED_FEATURE_COLUMNS if c in df.columns]
    df[feature_cols_present] = df[feature_cols_present].apply(pd.to_numeric, errors="coerce")
    df[feature_cols_present] = df[feature_cols_present].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols_present)
    df = df.drop_duplicates().reset_index(drop=True)
 
    print(f"  -> {df.shape[0]:,} rows kept")
    cleaned_pieces.append(df)
    del df
    gc.collect()
 
df = pd.concat(cleaned_pieces, ignore_index=True)
del cleaned_pieces
gc.collect()
 
feature_cols = [c for c in REDUCED_FEATURE_COLUMNS if c in df.columns]
constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
if constant_cols:
    df = df.drop(columns=constant_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]
 
print(f"Final feature count: {len(feature_cols)}")
 
BENIGN_LABEL = "Benign"
TRUE_UNSEEN_LABELS = ["DoS attacks-GoldenEye", "DoS attacks-Slowloris"]
 
benign_df = df[df["label"] == BENIGN_LABEL].copy()
unseen_df = df[df["label"].isin(TRUE_UNSEEN_LABELS)].copy()
del df
gc.collect()
 
print("Benign rows:", len(benign_df))
print("Unseen attack rows:", len(unseen_df))
 

#  SPLIT AND SCALE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
RANDOM_STATE = 42
 
X_benign = benign_df[feature_cols].values
X_train, X_temp = train_test_split(X_benign, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test_benign = train_test_split(X_temp, test_size=0.50, random_state=RANDOM_STATE)
X_test_unseen = unseen_df[feature_cols].values
 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_benign_s = scaler.transform(X_test_benign)
X_test_unseen_s = scaler.transform(X_test_unseen)
 
X_test_s = np.vstack([X_test_benign_s, X_test_unseen_s])
y_test = np.concatenate([np.zeros(len(X_test_benign_s)), np.ones(len(X_test_unseen_s))])
 
print("Train:", X_train_s.shape, "| Val:", X_val_s.shape)
 

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import roc_auc_score
 
tf.random.set_seed(RANDOM_STATE)
 
def build_autoencoder(input_dim, encoding_dim, l2_reg):
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
 
def reconstruction_error(model, X):
    X_pred = model.predict(X, batch_size=1024, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)
 
print("\n" + "#" * 60)
print("# HYPERPARAMETER SEARCH")
print("#" * 60)
 
# Subsample for search speed (same idea as the XGBoost 300k-row search subsample)
SEARCH_SAMPLE_SIZE = 300000
rng = np.random.RandomState(RANDOM_STATE)
search_idx = rng.choice(len(X_train_s), size=min(SEARCH_SAMPLE_SIZE, len(X_train_s)), replace=False)
X_train_search = X_train_s[search_idx]
 
GRID = [
    {"encoding_dim": 8,  "l2_reg": 1e-5},
    {"encoding_dim": 8,  "l2_reg": 1e-4},
    {"encoding_dim": 16, "l2_reg": 1e-5},
    {"encoding_dim": 16, "l2_reg": 1e-4},
    {"encoding_dim": 32, "l2_reg": 1e-5},
    {"encoding_dim": 32, "l2_reg": 1e-4},
]
 
input_dim = X_train_s.shape[1]
search_results = []
 
for i, params in enumerate(GRID):
    print(f"\n[{i+1}/{len(GRID)}] Testing encoding_dim={params['encoding_dim']}, l2_reg={params['l2_reg']}")
    start = time.time()
 
    model = build_autoencoder(input_dim, params["encoding_dim"], params["l2_reg"])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
 
    history = model.fit(
        X_train_search, X_train_search,
        validation_data=(X_val_s, X_val_s),
        epochs=40, batch_size=512, shuffle=True,
        callbacks=[early_stop, reduce_lr], verbose=0,
    )
 
    val_loss = min(history.history["val_loss"])
    elapsed = time.time() - start
 
    # Also compute unseen-attack ROC-AUC as a secondary, task-relevant metric
    val_errors = reconstruction_error(model, X_val_s)
    threshold = np.percentile(val_errors, 95)
    test_errors = reconstruction_error(model, X_test_s)
    auc = roc_auc_score(y_test, test_errors)
 
    print(f"  val_loss={val_loss:.6f} | unseen-attack ROC-AUC={auc:.4f} | {elapsed:.1f}s | {len(history.history['loss'])} epochs")
 
    search_results.append({
        "encoding_dim": params["encoding_dim"],
        "l2_reg": params["l2_reg"],
        "val_loss": val_loss,
        "roc_auc": auc,
        "epochs_run": len(history.history["loss"]),
        "seconds": elapsed,
    })
 
    del model
    tf.keras.backend.clear_session()
    gc.collect()
 
search_df = pd.DataFrame(search_results).sort_values("val_loss")
print("\n" + "=" * 60)
print("SEARCH RESULTS (sorted by validation loss)")
print("=" * 60)
print(search_df.to_string(index=False))
search_df.to_csv(f"{FIGURE_DIR}/autoencoder_tuning_search_results.csv", index=False)
 
best = search_df.iloc[0]
BEST_ENCODING_DIM = int(best["encoding_dim"])
BEST_L2_REG = float(best["l2_reg"])
print(f"\nBest configuration: encoding_dim={BEST_ENCODING_DIM}, l2_reg={BEST_L2_REG}")
 

print("\n" + "#" * 60)
print("# RETRAINING BEST CONFIGURATION ON FULL DATA")
print("#" * 60)
 
final_model = build_autoencoder(input_dim, BEST_ENCODING_DIM, BEST_L2_REG)
final_model.summary()
 
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
 
history = final_model.fit(
    X_train_s, X_train_s,
    validation_data=(X_val_s, X_val_s),
    epochs=100, batch_size=512, shuffle=True,
    callbacks=[early_stop, reduce_lr], verbose=1,
)
 
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5.5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title(f"Tuned Autoencoder Training (encoding_dim={BEST_ENCODING_DIM}, l2_reg={BEST_L2_REG})")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/tuned_training_validation_loss.png", dpi=150)
plt.show()
 
# ======================================================================
# SECTION: EVALUATE TUNED MODEL (same primary evaluation as before)
# ======================================================================
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
 
val_errors = reconstruction_error(final_model, X_val_s)
THRESHOLD = np.percentile(val_errors, 95)
print(f"\nTuned model threshold (95th pct): {THRESHOLD:.6f}")
 
test_errors = reconstruction_error(final_model, X_test_s)
y_pred = (test_errors > THRESHOLD).astype(int)
 
print("\n=== TUNED MODEL: Benign vs. True Unseen Attacks ===")
print(classification_report(y_test, y_pred, target_names=["Benign", "Unseen Attack"]))
tuned_auc = roc_auc_score(y_test, test_errors)
print("ROC-AUC:", tuned_auc)
 
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Unseen Attack"]).plot(
    cmap="Purples", values_format="d", ax=ax)
plt.title("Confusion Matrix -- Tuned Autoencoder")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/tuned_confusion_matrix.png", dpi=150)
plt.show()
 

import joblib
 
AE_TUNED_DIR = "C:/masters_thesis/artifacts/autoencoder_all4days_tuned"
os.makedirs(AE_TUNED_DIR, exist_ok=True)
 
final_model.save(f"{AE_TUNED_DIR}/autoencoder_model.keras")
joblib.dump(scaler, f"{AE_TUNED_DIR}/scaler.pkl")
joblib.dump(feature_cols, f"{AE_TUNED_DIR}/feature_columns.pkl")
joblib.dump(THRESHOLD, f"{AE_TUNED_DIR}/anomaly_threshold.pkl")
 
print("\nSaved tuned model to:", AE_TUNED_DIR)
print("(Original model at autoencoder_all4days is untouched.)")
 
print("\n" + "=" * 60)
print("COMPARISON: ORIGINAL vs. TUNED")
print("=" * 60)
print("Original model architecture: encoding_dim=16, l2_reg=1e-5 (manually chosen)")
print(f"Tuned model architecture:    encoding_dim={BEST_ENCODING_DIM}, l2_reg={BEST_L2_REG} (selected via search)")
print(f"Tuned model ROC-AUC: {tuned_auc:.4f}")
print("(Compare this to the original model's ROC-AUC of 0.9918 from your earlier run.)")
 
print("\nDONE.")
 