"""
Generate the synthetic datasets that ship with the course.

All datasets are produced deterministically (fixed seeds), so running this
script always recreates byte-identical CSV files.  Every generator documents
the *true* data-generating process, which is exactly what makes synthetic data
useful for teaching: we know the ground truth and can check whether our models
recover it.

    python make_datasets.py          # (re)creates the CSV files next to this script
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. Telco-style customer churn (tabular classification with messy data)
# ---------------------------------------------------------------------------
def make_customer_churn(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.40, 0.42, 0.18])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                         size=n, p=[0.34, 0.20, 0.23, 0.23])
    region = rng.choice(["North", "South", "East", "West"], size=n)
    senior = rng.random(n) < 0.16
    partner = rng.random(n) < 0.48
    tech_support = (rng.random(n) < 0.35) & (internet != "No")
    streaming = (rng.random(n) < 0.45) & (internet != "No")

    # tenure depends on contract type (longer contracts -> longer tenure)
    tenure = np.where(contract == "Month-to-month", rng.gamma(1.6, 12, n),
                      np.where(contract == "One year", rng.gamma(2.5, 14, n), rng.gamma(3.5, 16, n)))
    tenure = np.clip(np.round(tenure), 0, 72).astype(int)

    base = np.select([internet == "No", internet == "DSL", internet == "Fiber optic"], [20.0, 55.0, 85.0])
    monthly = base + 10 * tech_support + 12 * streaming + rng.normal(0, 4, n)
    monthly = np.round(np.clip(monthly, 15, 130), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    tickets = rng.poisson(0.6 + 1.2 * (internet == "Fiber optic") - 0.3 * tech_support, n)

    signup = pd.Timestamp("2024-06-30") - pd.to_timedelta(tenure * 30 + rng.integers(0, 30, n), unit="D")

    # --- the true churn mechanism (logit scale) -----------------------------
    logit = (-1.6
             + 1.3 * (contract == "Month-to-month")
             - 0.8 * (contract == "Two year")
             + 0.018 * (monthly - 65)
             - 0.035 * tenure
             + 0.9 * (tenure < 6)                     # early-life churn spike (non-linear)
             + 0.35 * tickets
             + 0.45 * (internet == "Fiber optic")
             - 0.6 * tech_support
             + 0.4 * senior
             + 0.5 * (payment == "Electronic check")
             - 0.25 * partner
             + 0.012 * tickets * (monthly - 65))      # interaction: expensive + unhappy
    churn = rng.random(n) < sigmoid(logit)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "region": region,
        "senior_citizen": senior.astype(int),
        "has_partner": partner.astype(int),
        "tenure_months": tenure,
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "tech_support": tech_support.astype(int),
        "streaming": streaming.astype(int),
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets": tickets,
        "churned": churn.astype(int),
    })

    # --- realistic data-quality problems (to be cleaned in the course) ------
    miss = rng.random(n) < 0.025
    df.loc[miss | (df["tenure_months"] == 0), "total_charges"] = np.nan
    messy = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[messy, "region"] = df.loc[messy, "region"].str.lower()
    outliers = rng.choice(n, size=3, replace=False)
    df.loc[outliers, "monthly_charges"] = 999.0
    dups = df.sample(5, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 2. Hourly electricity demand (time series with several seasonalities)
# ---------------------------------------------------------------------------
def make_energy_demand(start: str = "2022-01-01", end: str = "2023-12-31 23:00", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    t = np.arange(len(idx))
    hours = idx.hour.values
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values

    # temperature: yearly cycle + daily cycle + weather noise (AR(1))
    yearly = -np.cos(2 * np.pi * (doy - 15) / 365.25)
    daily = -np.cos(2 * np.pi * (hours - 3) / 24)
    weather = np.zeros(len(idx))
    eps = rng.normal(0, 0.9, len(idx))
    for i in range(1, len(idx)):
        weather[i] = 0.97 * weather[i - 1] + eps[i]
    temperature = 10 + 11 * yearly + 4 * daily + weather

    # demand: base + trend + daily double peak + weekend dip + temperature (U-shape) + noise
    trend = 1 + 0.02 * t / (24 * 365.25)
    daily_shape = (0.55 * np.exp(-((hours - 8) ** 2) / 8) + 1.0 * np.exp(-((hours - 18.5) ** 2) / 10)
                   - 0.35 * np.exp(-((hours - 3) ** 2) / 12))
    weekend = np.where(dow >= 5, -0.12, 0.0)
    temp_effect = 0.004 * np.clip(15 - temperature, 0, None) ** 1.5 + 0.012 * np.clip(temperature - 22, 0, None) ** 1.6
    holidays = pd.to_datetime(["2022-01-01", "2022-04-18", "2022-05-01", "2022-12-25", "2022-12-26",
                               "2023-01-01", "2023-04-10", "2023-05-01", "2023-12-25", "2023-12-26"])
    is_holiday = np.isin(idx.normalize(), holidays)
    noise = np.zeros(len(idx))
    e2 = rng.normal(0, 0.03, len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.8 * noise[i - 1] + e2[i]
    demand = 520 * trend * (1 + 0.25 * daily_shape + weekend - 0.15 * is_holiday + temp_effect + noise)
    df = pd.DataFrame({
        "timestamp": idx,
        "demand_mw": np.round(demand, 1),
        "temperature_c": np.round(temperature, 1),
        "is_holiday": is_holiday.astype(int),
    })
    return df


# ---------------------------------------------------------------------------
# 3. Loan applications with a documented historical bias (fairness notebook)
# ---------------------------------------------------------------------------
def make_loan_applications(n: int = 6000, seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.65, 0.35])
    age = np.clip(rng.normal(40, 12, n), 21, 75).round().astype(int)
    education = rng.choice(["high school", "bachelor", "master", "phd"], size=n, p=[0.35, 0.40, 0.20, 0.05])
    edu_num = pd.Series(education).map({"high school": 0, "bachelor": 1, "master": 2, "phd": 3}).values
    # group B has systematically lower income (a proxy for historical inequality)
    income = np.exp(rng.normal(10.6 + 0.15 * edu_num - 0.25 * (group == "B"), 0.45, n))
    income = np.round(income, -2)
    employment_years = np.clip(rng.gamma(2.0, 4.0, n), 0, 40).round(1)
    credit_history = np.clip(age - 21 - rng.gamma(2.0, 3.0, n), 0, None).round(1)
    debt = np.round(np.exp(rng.normal(8.5, 1.0, n)), -1)
    loan_amount = np.round(np.exp(rng.normal(9.6, 0.5, n)), -2)
    purpose = rng.choice(["car", "home improvement", "education", "business", "debt consolidation"],
                         size=n, p=[0.25, 0.20, 0.15, 0.15, 0.25])
    debt_ratio = debt / income
    # true ability to repay depends only on legitimate financial factors
    z_true = (1.2 + 0.9 * np.log(income / 40000) - 2.5 * debt_ratio + 0.05 * credit_history
              + 0.04 * employment_years - 0.6 * np.log(loan_amount / 15000) + 0.15 * edu_num)
    repaid = rng.random(n) < sigmoid(z_true)
    # historical approvals: the same factors PLUS a direct penalty for group B (label bias)
    z_hist = z_true - 1.1 * (group == "B") + rng.normal(0, 0.4, n)
    approved = rng.random(n) < sigmoid(z_hist)
    zip_prefix = np.where(group == "B", rng.choice([101, 102, 103], n), rng.choice([201, 202, 203, 204], n))
    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "age": age,
        "group": group,
        "education": education,
        "income": income,
        "employment_years": employment_years,
        "credit_history_years": credit_history,
        "existing_debt": debt,
        "loan_amount": loan_amount,
        "purpose": purpose,
        "zip_prefix": zip_prefix,
        "approved": approved.astype(int),
        "repaid": repaid.astype(int),
    })
    return df


# ---------------------------------------------------------------------------
# 4. Product reviews (small synthetic text corpus for the NLP notebook)
# ---------------------------------------------------------------------------
def make_reviews(n: int = 2400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products = {
        "headphones": ["sound", "bass", "battery", "comfort", "noise cancelling", "fit"],
        "laptop": ["keyboard", "screen", "battery life", "performance", "fan noise", "trackpad"],
        "coffee maker": ["coffee", "carafe", "brewing time", "cleaning", "temperature", "design"],
        "running shoes": ["cushioning", "grip", "sizing", "durability", "arch support", "weight"],
        "novel": ["plot", "characters", "pacing", "ending", "writing", "dialogue"],
        "vacuum": ["suction", "battery", "filter", "noise", "attachments", "weight"],
    }
    pos_adj = ["excellent", "great", "fantastic", "solid", "impressive", "wonderful", "reliable",
               "superb", "brilliant", "lovely", "outstanding", "smooth", "crisp", "comfortable"]
    neg_adj = ["terrible", "disappointing", "awful", "flimsy", "poor", "frustrating", "cheap",
               "useless", "mediocre", "dreadful", "noisy", "uncomfortable", "sluggish", "broken"]
    pos_templates = [
        "The {aspect} is {adj}.", "Really {adj} {aspect}, I love it.", "{Aspect} is {adj} and works as advertised.",
        "Absolutely {adj} — the {aspect} exceeded my expectations.", "Five stars for the {adj} {aspect}.",
        "I was skeptical but the {aspect} turned out {adj}.", "Not bad at all, the {aspect} is {adj}.",
        "Would buy again, {adj} {aspect} for the price.", "My favourite part is the {adj} {aspect}.",
    ]
    neg_templates = [
        "The {aspect} is {adj}.", "Honestly {adj} {aspect}, avoid.", "{Aspect} is {adj} and stopped working after a week.",
        "Not {padj} at all — the {aspect} is {adj}.", "One star, the {aspect} is simply {adj}.",
        "I expected more but the {aspect} is {adj}.", "Returned it, {adj} {aspect} and bad customer service.",
        "Nothing about the {aspect} is {padj}; it is {adj}.", "Sadly the {aspect} feels {adj}.",
    ]
    fillers = ["Shipping was fast.", "Packaging was fine.", "Bought it as a gift.", "Used it daily for a month.",
               "Price seems fair.", "Arrived on time.", "Colour matches the photos.", "Manual is in three languages."]

    rows = []
    for i in range(n):
        product = rng.choice(list(products))
        aspect = rng.choice(products[product])
        label = int(rng.random() < 0.5)
        n_sent = rng.integers(1, 4)
        sentences = []
        for _ in range(n_sent):
            if label == 1:
                s = rng.choice(pos_templates).format(aspect=aspect, Aspect=aspect.capitalize(), adj=rng.choice(pos_adj))
            else:
                s = rng.choice(neg_templates).format(aspect=aspect, Aspect=aspect.capitalize(),
                                                     adj=rng.choice(neg_adj), padj=rng.choice(pos_adj))
            sentences.append(s)
            aspect = rng.choice(products[product])
        if rng.random() < 0.6:
            sentences.insert(rng.integers(0, len(sentences) + 1), rng.choice(fillers))
        # a little label noise, as in real review data
        if rng.random() < 0.03:
            label = 1 - label
        rows.append({"review_id": i + 1, "product": product, "text": " ".join(sentences),
                     "rating": int(np.clip(rng.normal(4.3 if label else 1.8, 0.7), 1, 5).round()), "sentiment": label})
    return pd.DataFrame(rows)


def main() -> None:
    make_customer_churn().to_csv(HERE / "customer_churn.csv", index=False)
    make_energy_demand().to_csv(HERE / "energy_demand.csv", index=False)
    make_loan_applications().to_csv(HERE / "loan_applications.csv", index=False)
    make_reviews().to_csv(HERE / "reviews.csv", index=False)
    for f in sorted(HERE.glob("*.csv")):
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
