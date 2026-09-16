import time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
import shap
import streamlit as st

from tensorflow.keras import layers, models, regularizers

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
class XGBBoosterWrapper:

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self.booster.predict(dmat)
 
BASE_PATH ="."
 
XGB_DIR = f"{BASE_PATH}/artifacts/xgboost_no_leakage"
AE_DIR = f"{BASE_PATH}/artifacts/autoencoder_all4days"
HYBRID_DIR = f"{BASE_PATH}/artifacts/hybrid"
 
# Only the two validated Thursday files -- these match what the
# autoencoder was actually trained on, so results are trustworthy.
# (Contains: Benign, Infiltration, DoS-GoldenEye, DoS-Slowloris)
TEST_DATA_PATHS = [
    f"{BASE_PATH}/Data/sample/Thursday-15-02-2018_sample.csv",
    f"{BASE_PATH}/Data/sample/Thursday-01-03-2018_sample.csv",
    f"{BASE_PATH}/Data/sample/Friday-16-02-2018_sample.csv",
    f"{BASE_PATH}/Data/sample/Friday-02-03-2018_sample.csv",
]
 
ATTACK_SEVERITY_WEIGHT = {
    "Infilteration": 0.95,
    "DoS attacks-Hulk": 0.85,
    "Bot": 0.75,
    "DoS attacks-SlowHTTPTest": 0.65,
}
 
SEVERITY_COLORS = {
    "Critical": "#7f1d1d", "High": "#7c2d12",
    "Medium": "#78350f", "Low": "#052e16",
}
SEVERITY_TEXT_COLORS = {
    "Critical": "#fecaca", "High": "#fed7aa",
    "Medium": "#fde68a", "Low": "#86efac",
}
 
st.set_page_config(page_title="IDS Security Console", layout="wide")
 
st.markdown("""
<style>
@keyframes pulse-red {
  0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.7); }
  70%  { box-shadow: 0 0 0 18px rgba(220,38,38,0); }
  100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.emergency-banner {
  background: #7f1d1d; color: #fecaca; border: 2px solid #dc2626;
  border-radius: 8px; padding: 14px 22px; font-size: 17px; font-weight: 600;
  animation: pulse-red 1.4s infinite; margin-bottom: 12px;
}
.clear-banner {
  background: #052e16; color: #86efac; border: 2px solid #16a34a;
  border-radius: 8px; padding: 12px 22px; font-size: 15px; margin-bottom: 12px;
}
.live-dot {
  height: 10px; width: 10px; background-color: #16a34a; border-radius: 50%;
  display: inline-block; animation: pulse-dot 1.5s infinite; margin-right: 8px;
}
.event-row {
  display: grid; grid-template-columns: 90px 90px 90px 1fr 90px 80px;
  gap: 10px; padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;
  font-family: 'Courier New', monospace; font-size: 13px; align-items: center;
}
.event-header {
  display: grid; grid-template-columns: 90px 90px 90px 1fr 90px 80px;
  gap: 10px; padding: 6px 12px; font-size: 11px; color: #9ca3af;
  text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #374151;
  margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def load_artifacts():
    booster = xgb.Booster()
    booster.load_model(f"{XGB_DIR}/xgb_model.json")
    xgb_model = XGBBoosterWrapper(booster)
    label_encoder = joblib.load(f"{XGB_DIR}/label_encoder.pkl")
    xgb_feature_cols = joblib.load(f"{XGB_DIR}/feature_columns.pkl")
    xgb_scaler = joblib.load(f"{XGB_DIR}/scaler.pkl")
 
    ae_feature_cols = joblib.load(f"{AE_DIR}/feature_columns.pkl")
    autoencoder = build_autoencoder(input_dim=len(ae_feature_cols), encoding_dim=16)
    autoencoder.load_weights(f"{AE_DIR}/autoencoder_model.keras")
    ae_scaler = joblib.load(f"{AE_DIR}/scaler.pkl")
    anomaly_threshold = joblib.load(f"{AE_DIR}/anomaly_threshold.pkl")
 
    hybrid_config = joblib.load(f"{HYBRID_DIR}/hybrid_config.pkl")
    infil_threshold = hybrid_config["infil_confidence_threshold"]
 
    explainer = shap.TreeExplainer(booster)
 
    return {
        "xgb_model": xgb_model, "label_encoder": label_encoder,
        "xgb_feature_cols": xgb_feature_cols, "xgb_scaler": xgb_scaler,
        "autoencoder": autoencoder, "ae_scaler": ae_scaler,
        "ae_feature_cols": ae_feature_cols, "anomaly_threshold": anomaly_threshold,
        "infil_threshold": infil_threshold, "explainer": explainer,
    }
 
 
# CHANGED: removed src_ip, dst_ip, src_port -- CICIDS2018's flow CSVs
# never include these columns at all (unlike CICIDS2017). Only
# dst_port, protocol, timestamp, and label actually exist in this data.
DISPLAY_COLS_LOWER = {"label", "timestamp", "dst_port", "protocol"}
 
@st.cache_resource
def load_test_data(xgb_feature_cols, ae_feature_cols):
    needed_lower = set(xgb_feature_cols) | set(ae_feature_cols) | DISPLAY_COLS_LOWER
    pieces = []
    for path in TEST_DATA_PATHS:
        header = pd.read_csv(path, nrows=0)
        col_map = {c: c.strip().lower().replace(" ", "_") for c in header.columns}
        cols_to_keep = [orig for orig, low in col_map.items() if low in needed_lower]
        df = pd.read_csv(path, usecols=cols_to_keep, low_memory=False)
        pieces.append(df)
    combined = pd.concat(pieces, ignore_index=True)
    return combined
 
 
def preprocess_raw_flows(df_raw, xgb_feature_cols, ae_feature_cols):
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df[df["label"] != "Label"]
 
    if "timestamp" in df.columns:
        df["display_time"] = df["timestamp"]
        ts_parsed = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df["hour"] = ts_parsed.dt.hour
        df["day_of_week"] = ts_parsed.dt.dayofweek
        df = df.drop(columns=["timestamp"])
 
    required_cols = sorted(set(xgb_feature_cols) | set(ae_feature_cols))
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    df[required_cols] = df[required_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    return df
 
 
def predict_with_infil_threshold(y_proba, infil_idx, infil_min_conf):
    y_pred = np.argmax(y_proba, axis=1)
    infil_picked = (y_pred == infil_idx)
    low_conf = infil_picked & (y_proba[:, infil_idx] < infil_min_conf)
    if low_conf.any():
        proba_without_infil = y_proba.copy()
        proba_without_infil[:, infil_idx] = -1
        y_pred[low_conf] = np.argmax(proba_without_infil[low_conf], axis=1)
    return y_pred
 
 
def hybrid_predict_batch(df_raw, art):
    df = preprocess_raw_flows(df_raw, art["xgb_feature_cols"], art["ae_feature_cols"])
    if len(df) == 0:
        return df
 
    infil_idx = list(art["label_encoder"].classes_).index("Infilteration")
 
    X_xgb = df[art["xgb_feature_cols"]].values
    X_xgb_scaled = art["xgb_scaler"].transform(X_xgb)
    xgb_pred_proba = art["xgb_model"].predict_proba(X_xgb_scaled)
    xgb_pred_encoded = predict_with_infil_threshold(xgb_pred_proba, infil_idx, art["infil_threshold"])
    xgb_pred_labels = art["label_encoder"].inverse_transform(xgb_pred_encoded)
    xgb_confidence = xgb_pred_proba.max(axis=1)
 
    known_attack_mask = xgb_pred_labels != "Benign"
 
    recon_error = np.full(len(df), np.nan)
    benign_idx = np.where(~known_attack_mask)[0]
    if len(benign_idx) > 0:
        X_ae = df.iloc[benign_idx][art["ae_feature_cols"]].values
        X_ae_scaled = art["ae_scaler"].transform(X_ae)
        X_ae_pred = art["autoencoder"].predict(X_ae_scaled, batch_size=256, verbose=0)
        recon_error[benign_idx] = np.mean(np.square(X_ae_scaled - X_ae_pred), axis=1)
 
    final_verdict = []
    for i in range(len(df)):
        if known_attack_mask[i]:
            final_verdict.append(f"Known attack: {xgb_pred_labels[i]}")
        elif recon_error[i] > art["anomaly_threshold"]:
            final_verdict.append("Suspicious")
        else:
            final_verdict.append("Benign")
 
    results = df.copy()
    results["xgb_prediction"] = xgb_pred_labels
    results["xgb_confidence"] = xgb_confidence.round(4)
    results["reconstruction_error"] = recon_error
    results["final_verdict"] = final_verdict
    results["_X_xgb_scaled"] = list(X_xgb_scaled)
    return results
 
 
def compute_risk_score(row, anomaly_threshold):
    verdict = row["final_verdict"]
    confidence = row["xgb_confidence"]
    recon_error = row["reconstruction_error"]
 
    if verdict.startswith("Known attack"):
        attack_type = row["xgb_prediction"]
        severity_weight = ATTACK_SEVERITY_WEIGHT.get(attack_type, 0.7)
        score = 50 + 50 * confidence * severity_weight
    elif verdict == "Suspicious":
        ratio = recon_error / anomaly_threshold if anomaly_threshold > 0 else 1
        score = 30 + 55 * (1 - np.exp(-0.5 * (ratio - 1)))
        score = max(30, min(95, score))
    else:
        ratio = recon_error / anomaly_threshold if not np.isnan(recon_error) and anomaly_threshold > 0 else 0
        score = min(15, 15 * ratio)
    return round(score, 1)
 
 
def risk_level(score):
    if score >= 76:
        return "Critical"
    elif score >= 51:
        return "High"
    elif score >= 26:
        return "Medium"
    return "Low"
 
 
def explain_xgb_row(row, art):
    x_row = np.array(row["_X_xgb_scaled"]).reshape(1, -1)
    pred_label = row["xgb_prediction"]
    pred_class_idx = list(art["label_encoder"].classes_).index(pred_label)
    sv = art["explainer"].shap_values(x_row)
    row_shap = sv[0, :, pred_class_idx] if (not isinstance(sv, list) and sv.ndim == 3) else (sv[pred_class_idx][0] if isinstance(sv, list) else sv[0])
    explanation = pd.DataFrame({
        "feature": art["xgb_feature_cols"], "value": x_row[0], "contribution": row_shap
    }).sort_values("contribution", key=abs, ascending=False).head(5)
    return explanation
 
 
def explain_anomaly_row(row, art):
    x_raw = row[art["ae_feature_cols"]].values.astype(float).reshape(1, -1)
    x_scaled = art["ae_scaler"].transform(x_raw)
    x_recon = art["autoencoder"].predict(x_scaled, verbose=0)
    per_feature_error = np.square(x_scaled[0] - x_recon[0])
    breakdown = pd.DataFrame({
        "feature": art["ae_feature_cols"], "squared_error": per_feature_error
    }).sort_values("squared_error", ascending=False).head(5)
    return breakdown
 
 
def short_verdict(v):
    if v.startswith("Known attack"):
        return v.replace("Known attack: ", "")
    if v == "Suspicious":
        return "Unknown/Zero-day"
    return "Benign"
 
 
# ============================================================
# UI
# ============================================================
st.markdown(
    '<span class="live-dot"></span> **IDS Security Console** &nbsp;|&nbsp; '
    'Hybrid XGBoost + Autoencoder detection engine', unsafe_allow_html=True
)
st.write("")
 
with st.sidebar:
    st.header("Console controls")
    batch_size = st.slider("Flows per cycle", 5, 100, 20)
    delay = st.slider("Delay between cycles (seconds)", 0.5, 5.0, 1.5)
    col_a, col_b = st.columns(2)
    start_clicked = col_a.button("Start", type="primary")
    stop_clicked = col_b.button("Stop")
    st.caption("Replays real CICIDS2018 traffic through the validated hybrid pipeline.")
 
art = load_artifacts()
test_df = load_test_data(art["xgb_feature_cols"], art["ae_feature_cols"])
 
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "pointer" not in st.session_state:
    st.session_state.pointer = 0
if "all_results" not in st.session_state:
    st.session_state.all_results = pd.DataFrame()
 
if start_clicked:
    st.session_state.streaming = True
    st.session_state.pointer = 0
    st.session_state.all_results = pd.DataFrame()
if stop_clicked:
    st.session_state.streaming = False
 
banner = st.empty()
metric_cols = st.columns(3)
metric_placeholders = [c.empty() for c in metric_cols]
st.write("")
log_header = st.empty()
log_container = st.container()
if st.session_state.streaming:
    batch = test_df.sample(n=batch_size)
 
    batch_results = hybrid_predict_batch(batch, art)
    if len(batch_results) > 0:
        batch_results["risk_score"] = batch_results.apply(
            lambda r: compute_risk_score(r, art["anomaly_threshold"]), axis=1
        )
        batch_results["risk_level"] = batch_results["risk_score"].apply(risk_level)
 
        st.session_state.all_results = pd.concat(
            [st.session_state.all_results, batch_results], ignore_index=True
        ).tail(2000)
 
        batch_has_attack = (batch_results["final_verdict"] != "Benign").any()
        if batch_has_attack:
            n_attacks = (batch_results["final_verdict"] != "Benign").sum()
            banner.markdown(
                f'<div class="emergency-banner">&#9888; ACTIVE THREAT DETECTED &mdash; '
                f'{n_attacks} flagged flow(s) in this cycle</div>', unsafe_allow_html=True
            )
        else:
            banner.markdown(
                '<div class="clear-banner">All clear &mdash; latest cycle is benign</div>',
                unsafe_allow_html=True
            )
 
    all_results = st.session_state.all_results
    total = len(all_results)
    suspicious = (all_results["final_verdict"] == "Suspicious").sum()
    critical = (all_results["risk_level"] == "Critical").sum()
 
    metric_placeholders[0].metric("Flows analyzed", f"{total:,}")
    metric_placeholders[1].metric("Suspicious", f"{suspicious:,}")
    metric_placeholders[2].metric("Critical alerts", f"{critical:,}")
 
    log_header.subheader("Live event log")
    with log_container:
        # CHANGED: header + row grid now shows only fields that actually
        # exist in CICIDS2018 (Time, Dest Port, Protocol, Verdict, Risk,
        # Severity) instead of Source/Destination IP + Source Port,
        # which were always N/A because this dataset never has them.
        st.markdown(
            '<div class="event-header"><div>Time</div><div>Dst Port</div>'
            '<div>Proto</div><div>Verdict</div><div>Risk</div><div>Severity</div></div>',
            unsafe_allow_html=True
        )
        recent = all_results[all_results["final_verdict"] != "Benign"].tail(25).iloc[::-1]
        for idx, row in recent.iterrows():
            sev = row["risk_level"]
            bg = SEVERITY_COLORS[sev]
            fg = SEVERITY_TEXT_COLORS[sev]
            dst_port = row.get("dst_port", "-")
            proto = row.get("protocol", "-")
            t = row.get("display_time", "-")
            st.markdown(
                f'<div class="event-row" style="background:{bg}22;color:{fg};">'
                f'<div>{t}</div><div>{dst_port}</div><div>{proto}</div>'
                f'<div>{short_verdict(row["final_verdict"])}</div>'
                f'<div>{row["risk_score"]}</div><div>{sev}</div></div>', unsafe_allow_html=True
            )
            if row["final_verdict"] != "Benign":
                with st.expander("Details", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.write(f"**Confidence:** {row['xgb_confidence']:.2%}")
                    if not np.isnan(row["reconstruction_error"]):
                        c2.write(f"**Reconstruction error:** {row['reconstruction_error']:.4f}")
 
                    if row["final_verdict"].startswith("Known attack"):
                        st.write("**Top contributing features (SHAP):**")
                        st.dataframe(explain_xgb_row(row, art), hide_index=True, use_container_width=True)
                    else:
                        st.write("**Top features driving the anomaly:**")
                        st.dataframe(explain_anomaly_row(row, art), hide_index=True, use_container_width=True)
 
    time.sleep(delay)
    st.rerun()
else:
    st.info("Click **Start** in the sidebar to begin live monitoring.")
 





























































