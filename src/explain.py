"""
explain.py
----------
SHAP explainability utilities for the churn model.

We compute SHAP values in the model's PROCESSED feature space (i.e. after
the ColumnTransformer's OneHotEncoder + StandardScaler), not the raw mixed
dtype space. This is the robust, standard approach for SHAP with sklearn
pipelines: it avoids dtype issues from mixing numeric and categorical raw
columns, and `preprocessor.get_feature_names_out()` gives human-readable
names for every resulting (one-hot) column.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import json


def load_artifacts(models_dir="models"):
    preprocessor = joblib.load(f"{models_dir}/preprocessor.pkl")
    model = joblib.load(f"{models_dir}/churn_model.pkl")
    with open(f"{models_dir}/feature_names.json") as f:
        feature_names = json.load(f)
    background = pd.read_csv(f"{models_dir}/shap_background.csv")
    return preprocessor, model, feature_names, background


def build_explainer(preprocessor, model, background_raw, n_background=50):
    """Builds a SHAP Explainer over the model's probability output, operating
    in the PROCESSED (numeric, one-hot encoded) feature space."""
    bg_sample = background_raw.sample(min(n_background, len(background_raw)), random_state=42)
    bg_processed = preprocessor.transform(bg_sample)
    if hasattr(bg_processed, "toarray"):
        bg_processed = bg_processed.toarray()

    def predict_fn(X_proc):
        return model.predict_proba(X_proc)[:, 1]

    explainer = shap.Explainer(predict_fn, bg_processed)
    return explainer


def explain_instance(explainer, preprocessor, instance_df):
    """Returns SHAP values for a single-row (or multi-row) raw dataframe."""
    instance_proc = preprocessor.transform(instance_df)
    if hasattr(instance_proc, "toarray"):
        instance_proc = instance_proc.toarray()
    shap_values = explainer(instance_proc)
    return shap_values


def top_factors(shap_values, feature_names, top_n=5):
    """Return the top N features pushing the prediction up (risk) and
    down (retention) for a single instance."""
    values = shap_values.values[0]
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: x[1], reverse=True)

    increasing = [p for p in pairs if p[1] > 0][:top_n]
    decreasing = [p for p in pairs if p[1] < 0][:top_n]
    return increasing, decreasing

