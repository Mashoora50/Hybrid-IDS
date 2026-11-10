# Explainable Hybrid Intrusion Detection and Cyber Risk Framework
It combines two models:

- **XGBoost** recognises attacks it was trained on (Bot, DoS-Hulk, DoS-SlowHTTPTest, Infiltration).
- **Autoencoder** learns what normal traffic looks like and flags anything unusual, including attacks it has never seen.

Every alert gets a **risk score (0-100)** and a plain-language explanation of which traffic features triggered it.

---

## How it works

```
Network flow
    |
    v
XGBoost classifier ----> Known attack? --> labelled (e.g. "DoS attacks-Hulk")
    |
    | (called Benign)
    v
Autoencoder ----> Reconstruction error above threshold? --> "Suspicious / unknown"
    |
    | (below threshold)
    v
Benign

Every flagged flow --> Risk score (0-100) + explanation
```

| Part | What it does |
|---|---|
| XGBoost | Names known attack types |
| Autoencoder | Catches unfamiliar traffic (possible zero-day attacks) |
| SHAP | Explains XGBoost decisions feature by feature |
| Reconstruction-error breakdown | Explains autoencoder alerts (which features looked abnormal) |
| Risk engine | Combines model confidence, attack severity and anomaly strength into Low / Medium / High / Critical |
| Streamlit dashboard | Live-style monitoring console with colour-coded alerts |

---

## Dataset

[CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) (Canadian Institute for Cybersecurity). Four daily files are used:

- Thursday-15-02-2018
- Thursday-01-03-2018
- Friday-16-02-2018
- Friday-02-03-2018

The raw CSV files are not included in this repository because of their size. Download them and place them in `Data/raw/`.

---

## Key findings

**1. Hidden data leakage, found with SHAP.**
The first XGBoost model scored a macro F1 of 0.95, but SHAP showed it relied mostly on `day_of_week` and `hour`. In CICIDS2018 each attack happens on specific days, so the model was using the date as a shortcut instead of learning attack behaviour. Both features were removed and the model retrained.

**2. Honest results after the fix.**

| Model | Result |
|---|---|
| XGBoost (no leakage) | Accuracy 0.965, macro F1 0.891 |
| Bot, DoS-Hulk, DoS-SlowHTTPTest | F1 1.00 (unaffected by removing the leaky features) |
| Infiltration | F1 0.48 (hard to tell apart from normal traffic) |
| Autoencoder vs unseen attacks (GoldenEye, Slowloris) | ROC-AUC 0.977, recall 0.82, precision 0.71 |
| Autoencoder on normal traffic | 95% correctly passed as benign |

Removing the leakage also cut false alarms on normal traffic sharply (about 30% down to about 8% in the hybrid test on the Thursday files).

**3. Training data variety matters.**
An autoencoder trained on two days over-flagged normal traffic from other days. Retraining on benign traffic from all four days fixed this with almost no loss in attack detection.

**4. Infiltration is the weak spot.**
It is built to look like normal traffic, so both models struggle with it. This is why the system uses two layers rather than one model.

---

## Project structure

```
masters_thesis/
|-- app.py                     Streamlit dashboard
|-- hybrid_pipeline.py         Detection pipeline (XGBoost + autoencoder)
|-- retrain_autoencoder.py     Autoencoder training on all four days
|-- artifacts/
|   |-- xgboost_no_leakage/    Final XGBoost model and scaler
|   |-- autoencoder_all4days/  Final autoencoder, scaler, threshold
|   |-- hybrid/                Saved settings (confidence threshold)
|   `-- archive/               Earlier versions kept for comparison
|-- Data/
|   |-- raw/                   CICIDS2018 CSV files (not in repo)
|   `-- processed/             Exported results
|-- notebook/                  Jupyter notebooks (XGBoost, autoencoder, hybrid, SHAP and risk scoring)
`-- docs/                      Proposal and write-ups
```


🚀 Live Application

The Hybrid Intrusion Detection System (Hybrid IDS) is deployed using Streamlit.

Try the live application here:https://hybrid-ids-6ljkckappau9i4zb5qelofd.streamlit.app/

Open Hybrid IDS Application

## The dashboard

- Replays real CICIDS2018 traffic in small random batches, so it looks like live monitoring.
- Red alert banner when a batch contains attacks.
- Event log with colour-coded severity (Critical, High, Medium, Low).
- A **Details** panel on each alert shows the explanation: SHAP for known attacks, feature-level anomaly breakdown for suspicious flows.

---

## Limitations

- The dashboard replays recorded data. It does not capture live network traffic, and Apache Kafka was not used.
- Infiltration detection is limited (see key findings).
- The CICIDS2018 export used here has no source or destination IP columns, so the log shows ports and protocol only.
- Risk-score severity weights per attack type are reasonable assumptions, not values validated against real incident data.
- Results come from one dataset family (CICIDS2018). A test on CIC-UNSW-NB15 performed poorly and is not part of the final results.

---

## Tech stack

Python, XGBoost, TensorFlow/Keras, SHAP, scikit-learn, pandas, NumPy, Streamlit

---

## Author

Mashoora, MSc Data Science and AI, Middlesex University Dubai




