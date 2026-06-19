# 📡 Customer Churn Prediction System (Telecom)

An end-to-end machine learning system that predicts which telecom customers
are likely to churn, explains *why* using SHAP, and presents everything in
a polished, custom-styled Streamlit dashboard.

## ✨ What's inside

| Component | Details |
|---|---|
| **Data prep** | Missing-value handling, dedup, type fixes (`src/preprocessing.py`) |
| **Feature engineering** | 12 engineered features: tenure buckets, value tiers, add-on counts, risk flags, etc. |
| **Class balancing** | SMOTE oversampling applied only to the training split (no leakage) |
| **Model** | **Stacking ensemble**: Random Forest + XGBoost + LightGBM → Logistic Regression meta-learner |
| **Explainability** | SHAP (global feature importance + per-customer waterfall plots) |
| **Dashboard** | 5-page Streamlit app with custom HTML/CSS dark theme |

### Why a stacking ensemble instead of just Random Forest or XGBoost?

Single models each capture different patterns in the data. Stacking lets a
meta-learner (Logistic Regression) learn the optimal way to *combine* the
three base models' probability outputs, which typically beats any single
base model by 2-5 points of ROC-AUC on this type of tabular, imbalanced
dataset — without the instability that comes from over-tuning one model.

## 📁 Project Structure

```
churn_project/
├── app.py                          # Streamlit dashboard (run this)
├── requirements.txt
├── README.md
├── data/
│   ├── generate_sample_data.py     # Creates a synthetic demo dataset
│   └── telco_churn_synthetic.csv   # Generated demo data (already included)
├── src/
│   ├── preprocessing.py            # Cleaning + feature engineering
│   ├── train_model.py              # Full training pipeline
│   └── explain.py                  # SHAP helper functions
└── models/                         # Saved model artifacts (already included)
    ├── preprocessor.pkl
    ├── churn_model.pkl
    ├── metrics.json
    ├── feature_names.json
    ├── holdout_sample.csv
    └── shap_background.csv
```

## 🚀 Quick Start (already trained — just run the app)

This project ships with a **pre-trained model** so you can run the dashboard
immediately:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔁 Retraining (recommended: use the REAL Kaggle dataset)

The bundled model was trained on a synthetic dataset (generated to mimic the
real IBM Telco dataset's schema and statistical patterns), because the real
file requires a Kaggle account to download. **For best accuracy, retrain on
the real data:**

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Rename it to `telco_churn.csv` and place it in the `data/` folder.
3. Retrain:
   ```bash
   python src/train_model.py
   ```
   The script automatically detects and prefers `data/telco_churn.csv` over
   the synthetic file.
4. Relaunch the dashboard — it will pick up the newly trained model:
   ```bash
   streamlit run app.py
   ```

To regenerate the synthetic dataset instead (e.g. for testing):
```bash
python data/generate_sample_data.py
```

## 📊 Current Model Performance (synthetic demo data)

See `models/metrics.json` for the latest numbers, or check the **Model
Performance** tab in the dashboard. Typical results on the real Kaggle
dataset for this architecture are **~80-82% accuracy and ~0.84-0.86
ROC-AUC**, ahead of single-model baselines (plain Random Forest or
Logistic Regression typically land at ~79-80% accuracy / ~0.83 ROC-AUC
on this dataset).

## 🖥️ Dashboard Pages

1. **📊 Overview & EDA** — KPIs and churn-rate breakdowns by contract,
   internet service, payment method, and tenure.
2. **🤖 Model Performance** — Accuracy/Precision/Recall/F1/ROC-AUC, cross-
   validation results, and global SHAP feature importance.
3. **🔮 Predict Single Customer** — Interactive form to score one customer,
   with a risk badge and a SHAP explanation of which factors pushed the
   prediction up or down.
4. **📁 Batch Prediction (CSV)** — Upload a CSV of customers and get back a
   scored, downloadable file with churn probabilities and risk tiers.
5. **💡 Business Insights** — Plain-English findings and recommended
   retention actions for the business team.

## ☁️ Deploying to Streamlit Community Cloud

1. Push this whole folder to a **public GitHub repository**.
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app.
5. (Optional) If you retrained on the real dataset locally, make sure the
   updated `models/*.pkl` files are committed and pushed too — the deployed
   app uses whatever is in the repo.

> **Note:** Keep model files under GitHub's 100MB file size limit (they are
> well under it by default — typically a few MB).

## 🛠️ Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` ·
`imbalanced-learn (SMOTE)` · `SHAP` · `Streamlit` · `matplotlib`

## 📌 Notes & Limitations

- The bundled dataset is **synthetically generated** to match the real
  dataset's schema and realistic churn-driver relationships (contract type,
  payment method, tenure, internet service, etc.), since the real Kaggle
  file requires authentication to download programmatically. Retrain on the
  real file for production-grade numbers (see "Retraining" above).
- SHAP values are computed using a Permutation explainer over the model's
  processed (one-hot encoded) feature space for compatibility with the
  stacking ensemble.
