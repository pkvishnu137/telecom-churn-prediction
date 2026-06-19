"""
generate_sample_data.py
------------------------
Generates a synthetic dataset that EXACTLY matches the schema of the
IBM Telco Customer Churn dataset (Kaggle: blastchar/telco-customer-churn).

WHY THIS EXISTS:
The real dataset must be downloaded manually from Kaggle (requires a Kaggle
login), so this script lets you run and test the entire pipeline immediately.

>>> TO USE THE REAL DATASET INSTEAD <<<
1. Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" from:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Place it at: data/telco_churn.csv
3. Skip running this script. The training pipeline (src/train_model.py)
   will automatically use the real file if it exists, and only falls
   back to generating synthetic data if it doesn't.

Run:
    python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 7043  # same size as the real IBM Telco dataset

def generate():
    customer_id = [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(N)]

    gender = np.random.choice(["Male", "Female"], N)
    senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])

    tenure = np.random.gamma(shape=2.0, scale=16, size=N).astype(int)
    tenure = np.clip(tenure, 0, 72)

    phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
    multiple_lines = []
    for p in phone_service:
        if p == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.42, 0.58]))

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

    def dependent_internet_choice(p_yes=0.4):
        out = []
        for s in internet_service:
            if s == "No":
                out.append("No internet service")
            else:
                out.append(np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes]))
        return out

    online_security = dependent_internet_choice(0.35)
    online_backup = dependent_internet_choice(0.40)
    device_protection = dependent_internet_choice(0.40)
    tech_support = dependent_internet_choice(0.35)
    streaming_tv = dependent_internet_choice(0.45)
    streaming_movies = dependent_internet_choice(0.45)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges depend on services selected (more services -> higher cost)
    base = 18.0
    monthly_charges = []
    for i in range(N):
        charge = base
        if internet_service[i] == "DSL":
            charge += np.random.uniform(15, 30)
        elif internet_service[i] == "Fiber optic":
            charge += np.random.uniform(35, 60)
        if phone_service[i] == "Yes":
            charge += np.random.uniform(5, 15)
        for svc in [online_security[i], online_backup[i], device_protection[i],
                    tech_support[i], streaming_tv[i], streaming_movies[i]]:
            if svc == "Yes":
                charge += np.random.uniform(3, 8)
        monthly_charges.append(round(charge + np.random.normal(0, 3), 2))
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, N), 2)
    total_charges = np.clip(total_charges, 0, None)
    # a few blanks like the real dataset (tenure==0 customers)
    total_charges = total_charges.astype(object)
    total_charges[tenure == 0] = " "

    # ---- Churn probability model (creates realistic, learnable signal) ----
    churn_prob = np.full(N, 0.05)
    churn_prob += np.where(contract == "Month-to-month", 0.30, 0.0)
    churn_prob += np.where(contract == "One year", 0.05, 0.0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.15, 0.0)
    churn_prob += np.where(payment_method == "Electronic check", 0.12, 0.0)
    churn_prob += np.where(np.array(tech_support) == "No", 0.08, 0.0)
    churn_prob += np.where(np.array(online_security) == "No", 0.06, 0.0)
    churn_prob += np.where(senior_citizen == 1, 0.07, 0.0)
    churn_prob += np.where(partner == "No", 0.05, 0.0)
    churn_prob += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.06
    churn_prob -= (tenure / 72) * 0.35
    churn_prob += np.random.normal(0, 0.05, N)
    churn_prob = np.clip(churn_prob, 0.02, 0.95)

    churn = np.array(["Yes" if np.random.rand() < p else "No" for p in churn_prob])

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = os.path.join(os.path.dirname(__file__), "telco_churn_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn']=='Yes').mean():.2%}")
