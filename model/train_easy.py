# data/generate_dataset.py
"""
Generates a realistic synthetic microcredit dataset for Indian young borrowers.
Features simulate: income, mobile usage, education, employment, etc.
"""

import pandas as pd
import numpy as np
import os

def generate_microcredit_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)

    data = {}

    # === DEMOGRAPHICS ===
    data['age'] = np.random.randint(18, 45, n_samples)
    data['gender'] = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])  # 0=F, 1=M
    data['education_level'] = np.random.choice(
        [1, 2, 3, 4], n_samples, p=[0.15, 0.35, 0.35, 0.15]
    )  # 1=below_HS, 2=HS, 3=graduate, 4=postgrad
    data['marital_status'] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.5, 0.1])
    # 0=single, 1=married, 2=divorced
    data['dependents'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    data['city_tier'] = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
    # 1=metro, 2=tier2, 3=rural

    # === FINANCIAL ===
    base_income = 8000 + data['education_level'] * 7000 + data['age'] * 300
    data['monthly_income'] = (base_income + np.random.normal(0, 5000, n_samples)).clip(5000, 150000).astype(int)
    data['monthly_expenses'] = (data['monthly_income'] * np.random.uniform(0.3, 0.9, n_samples)).astype(int)
    data['existing_emi'] = (data['monthly_income'] * np.random.uniform(0, 0.4, n_samples) * 
                            np.random.choice([0, 1], n_samples, p=[0.4, 0.6])).astype(int)
    data['savings_balance'] = (np.random.exponential(10000, n_samples) + 
                                data['monthly_income'] * np.random.uniform(0, 3, n_samples)).astype(int)
    data['bank_account_age_months'] = np.random.randint(0, 120, n_samples)

    # === LOAN REQUEST ===
    data['loan_amount_requested'] = np.random.choice(
        [5000, 10000, 15000, 20000, 25000, 30000, 50000], n_samples
    )
    data['loan_tenure_months'] = np.random.choice([3, 6, 9, 12, 18, 24], n_samples)
    data['loan_purpose'] = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )  # 1=education, 2=medical, 3=business, 4=personal, 5=other

    # === DIGITAL FOOTPRINT (simulated mobile/UPI signals) ===
    data['mobile_verified'] = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    data['upi_transactions_monthly'] = np.random.poisson(15, n_samples)
    data['avg_upi_amount'] = (np.random.exponential(500, n_samples) + 100).astype(int)
    data['social_media_score'] = np.random.randint(0, 100, n_samples)
    data['phone_os'] = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])  # 0=iOS, 1=Android

    # === CREDIT HISTORY (thin-file signals) ===
    data['prev_loans_count'] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, 
                                                  p=[0.35, 0.25, 0.2, 0.1, 0.06, 0.04])
    data['prev_defaults'] = np.where(
        data['prev_loans_count'] > 0,
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        0
    )
    data['credit_utilization_pct'] = np.random.uniform(0, 100, n_samples).round(1)

    # === ENGINEERED FEATURES ===
    data['debt_to_income'] = np.where(
        data['monthly_income'] > 0,
        ((data['existing_emi'] + data['loan_amount_requested'] / data['loan_tenure_months']) / 
         data['monthly_income'] * 100).round(2),
        100.0
    )
    data['disposable_income'] = data['monthly_income'] - data['monthly_expenses'] - data['existing_emi']
    data['loan_to_income_ratio'] = (data['loan_amount_requested'] / data['monthly_income']).round(3)

    # === TARGET: credit_worthy (1=approve, 0=reject) ===
    # Realistic scoring logic
    score = np.zeros(n_samples, dtype=float)
    score += (data['monthly_income'] - 15000) / 10000 * 1.5
    score += (data['education_level'] - 2) * 0.8
    score -= data['prev_defaults'] * 3.0
    score += data['prev_loans_count'] * 0.3  # some history is good
    score -= (data['debt_to_income'] - 30) / 20 * 1.5
    score += data['disposable_income'] / 10000 * 1.0
    score += data['upi_transactions_monthly'] / 20 * 0.5
    score += data['bank_account_age_months'] / 60 * 0.8
    score -= (data['loan_amount_requested'] - 15000) / 10000 * 0.5
    score += data['mobile_verified'] * 0.5
    score += data['savings_balance'] / 20000 * 0.5
    score -= data['dependents'] * 0.3
    score += np.random.normal(0, 1.2, n_samples)  # noise

    prob = 1 / (1 + np.exp(-score))
    data['credit_worthy'] = (prob > 0.5).astype(int)

    df = pd.DataFrame(data)

    # Add applicant IDs (pseudo-anonymized)
    df.insert(0, 'applicant_id', [f"APP{str(i).zfill(6)}" for i in range(n_samples)])

    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/microcredit_dataset.csv', index=False)

    print(f"Dataset generated: {df.shape}")
    print(f"Class distribution:\n{df['credit_worthy'].value_counts(normalize=True).round(3)}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nSaved to data/microcredit_dataset.csv")

    return df


# === Option B: Download UCI German Credit / Lending Club ===
def download_uci_credit():
    """Download and prep UCI German Credit dataset as fallback."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_status', 'employment', 'installment_rate', 'personal_status',
        'other_parties', 'residence_since', 'property_magnitude', 'age',
        'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker', 'class'
    ]
    try:
        df = pd.read_csv(url, sep=' ', header=None, names=columns)
        df['class'] = df['class'].map({1: 1, 2: 0})  # 1=good, 2=bad -> 0=bad
        df.to_csv('data/uci_german_credit.csv', index=False)
        print(f"UCI German Credit: {df.shape}")
        return df
    except Exception as e:
        print(f"Could not download UCI data: {e}")
        return None


if __name__ == "__main__":
    df = generate_microcredit_dataset(n_samples=10000)