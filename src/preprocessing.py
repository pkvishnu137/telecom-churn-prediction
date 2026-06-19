"""
preprocessing.py
-----------------
Cleaning, missing-value handling, and feature engineering for the
Telco Customer Churn dataset.
"""

import numpy as np
import pandas as pd


SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies"
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values & type fixes (matches known issues in the
    real IBM Telco dataset, e.g. blank strings in TotalCharges)."""
    df = df.copy()

    # TotalCharges sometimes arrives as blank strings -> coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Customers with 0 tenure have NaN TotalCharges -> fill with 0
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop identifier column for modeling later (kept aside in pipeline)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new, informative features for the model."""
    df = df.copy()

    # 1. Tenure groups (bucketed customer lifecycle stage)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"]
    )

    # 2. Average monthly spend over the customer's lifetime
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # 3. Spend deviation: is the customer paying more than their historical average?
    df["ChargeDeviation"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]

    # 4. Count of subscribed add-on services (signal: engaged customers churn less)
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumAddOnServices"] = df[addon_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # 5. Has internet but no security/tech-support add-ons (high-risk segment)
    df["NoProtection"] = (
        (df["InternetService"] != "No") &
        (df["OnlineSecurity"] != "Yes") &
        (df["TechSupport"] != "Yes")
    ).astype(int)

    # 6. Total number of services subscribed to (phone + internet + addons)
    df["TotalServices"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int) +
        df["NumAddOnServices"]
    )

    # 7. Contract risk flag (month-to-month is the single biggest churn driver)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    # 8. Payment risk flag (electronic check correlates strongly with churn)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # 9. Family status combined
    df["HasFamilySupport"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

    # 10. Customer value tier (helps the model and dashboard segment customers).
    # Fixed bin edges (rather than qcut) so this works identically for batch
    # training data AND single-row inference at prediction time.
    df["ValueTier"] = pd.cut(
        df["MonthlyCharges"], bins=[-1, 45, 75, 1000], labels=["Low", "Medium", "High"]
    )

    # 11. New customer flag (high churn-risk early window)
    df["IsNewCustomer"] = (df["tenure"] <= 3).astype(int)

    # 12. Fiber optic without streaming bundle (price-sensitive disengaged segment)
    df["FiberNoStreaming"] = (
        (df["InternetService"] == "Fiber optic") &
        (df["StreamingTV"] != "Yes") &
        (df["StreamingMovies"] != "Yes")
    ).astype(int)

    return df


def full_pipeline(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = full_pipeline("data/telco_churn_synthetic.csv")
    print(df.shape)
    print(df.head())
