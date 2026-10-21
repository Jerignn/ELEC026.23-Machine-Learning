"""
generate_data.py
----------------
Creates the sample datasets for the ShopSmart Analytics project.

Run once (app.py also runs it automatically if the data folder is missing):
    python generate_data.py

Outputs (in ./data):
    products.csv    - product catalog
    sales.csv       - 24 months of monthly sales per product   (Linear Regression)
    reviews.csv     - customer reviews labeled pos/neg          (Naive Bayes)
    ratings.csv     - customer x product ratings                (KNN)
    customers.csv   - customer profile + behavior + labels      (SVM, Decision Tree, ANN)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = np.random.default_rng(42)

PRODUCTS = [
    # id, name, category, price
    (1, "Wireless Earbuds", "Electronics", 59.99),
    (2, "Smart Watch", "Electronics", 149.99),
    (3, "Bluetooth Speaker", "Electronics", 39.99),
    (4, "Phone Charger", "Electronics", 14.99),
    (5, "Running Shoes", "Sports", 89.99),
    (6, "Yoga Mat", "Sports", 24.99),
    (7, "Water Bottle", "Sports", 12.99),
    (8, "Dumbbell Set", "Sports", 69.99),
    (9, "Coffee Maker", "Home", 79.99),
    (10, "Air Fryer", "Home", 99.99),
    (11, "Desk Lamp", "Home", 29.99),
    (12, "Throw Pillow", "Home", 19.99),
    (13, "Face Moisturizer", "Beauty", 22.99),
    (14, "Hair Dryer", "Beauty", 44.99),
    (15, "Lipstick Set", "Beauty", 18.99),
    (16, "Perfume", "Beauty", 64.99),
    (17, "Backpack", "Fashion", 49.99),
    (18, "Sunglasses", "Fashion", 34.99),
    (19, "Denim Jacket", "Fashion", 74.99),
    (20, "Leather Wallet", "Fashion", 27.99),
]
CATEGORIES = ["Electronics", "Sports", "Home", "Beauty", "Fashion"]


def make_products():
    return pd.DataFrame(PRODUCTS, columns=["product_id", "name", "category", "price"])


def make_sales(products):
    """Monthly units sold for 24 months: base level + growth trend + seasonality
    (Nov/Dec holiday peak) + marketing spend effect + noise."""
    months = pd.date_range("2024-11-01", periods=24, freq="MS")
    rows = []
    for _, p in products.iterrows():
        base = rng.uniform(150, 600) * (60 / (p.price + 20))  # cheaper items sell more
        base = max(base, 40)
        growth = rng.uniform(0.005, 0.03)  # monthly growth rate
        for t, m in enumerate(months):
            ad_spend = round(rng.uniform(500, 3000), 2)
            season = 1 + 0.35 * (m.month in (11, 12)) + 0.10 * (m.month in (6, 7))
            units = base * (1 + growth * t) * season + 0.04 * ad_spend + rng.normal(0, base * 0.06)
            units = int(max(units, 5))
            rows.append({
                "month": m.strftime("%Y-%m"),
                "t": t,
                "month_num": m.month,
                "product_id": p.product_id,
                "ad_spend": ad_spend,
                "units_sold": units,
                "revenue": round(units * p.price, 2),
            })
    return pd.DataFrame(rows)


POS_PHRASES = [
    "love this product", "works perfectly", "excellent quality", "great value for money",
    "fast delivery", "highly recommend", "very comfortable", "exceeded my expectations",
    "amazing sound", "beautiful design", "easy to use", "would buy again",
    "super happy with it", "five stars", "arrived on time and well packed",
    "my family loves it", "battery lasts long", "fits perfectly", "great customer service",
    "smells wonderful", "sturdy and reliable",
]
NEG_PHRASES = [
    "broke after a week", "terrible quality", "waste of money", "arrived late",
    "do not recommend", "very uncomfortable", "not as described", "stopped working",
    "poor customer service", "cheap material", "returned it", "disappointed",
    "package was damaged", "battery dies quickly", "wrong size", "too expensive for what it is",
    "smells awful", "fell apart", "never buying again", "one star", "flimsy and noisy",
]
NEUTRAL_FILLER = [
    "bought this last month", "ordered for my sister", "the color is blue",
    "it came in a box", "used it a few times", "got it on sale", "compared with other brands",
]


def make_reviews(products, n=800):
    rows = []
    for i in range(n):
        label = "positive" if rng.random() < 0.62 else "negative"
        main = POS_PHRASES if label == "positive" else NEG_PHRASES
        other = NEG_PHRASES if label == "positive" else POS_PHRASES
        parts = list(rng.choice(main, size=rng.integers(1, 4), replace=False))
        if rng.random() < 0.15:  # a little noise: one opposite phrase
            parts.append(str(rng.choice(other)))
        if rng.random() < 0.5:
            parts.append(str(rng.choice(NEUTRAL_FILLER)))
        rng.shuffle(parts)
        text = ". ".join(parts).capitalize() + "."
        pid = int(rng.integers(1, len(products) + 1))
        rows.append({"review_id": i + 1, "product_id": pid, "text": text, "sentiment": label})
    return pd.DataFrame(rows)


def make_customers(n=1200):
    """Customer profile, browsing behavior and three target labels."""
    age = rng.integers(18, 70, n)
    income = np.round(rng.normal(45000, 15000, n).clip(12000, 120000), -2)
    tenure_months = rng.integers(1, 60, n)
    past_purchases = rng.poisson(4, n)
    sessions_per_month = rng.poisson(6, n) + 1
    avg_session_min = np.round(rng.gamma(2.5, 3, n), 1)
    pages_per_session = np.round(rng.gamma(3, 2, n), 1)
    cart_adds = rng.poisson(2, n)
    wishlist_items = rng.poisson(1.5, n)
    days_since_last_visit = rng.integers(0, 60, n)
    email_open_rate = np.round(rng.beta(2, 3, n), 2)
    used_coupon_before = rng.integers(0, 2, n)
    favorite_category = rng.choice(CATEGORIES, n)

    # --- will_purchase (SVM target + ANN target): driven by history & behavior
    z = (-3.2 + 0.25 * past_purchases + 0.18 * sessions_per_month + 0.45 * cart_adds
         + 0.08 * avg_session_min - 0.05 * days_since_last_visit + 0.00001 * income
         + 0.2 * wishlist_items + rng.normal(0, 0.8, n))
    will_purchase = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

    # --- responded_to_promo (Decision Tree target): rule-like behavior
    promo_score = (
        (email_open_rate > 0.45).astype(int) * 2
        + used_coupon_before * 2
        + (age < 35).astype(int)
        + (days_since_last_visit < 14).astype(int)
        + (income < 40000).astype(int)
    )
    responded = (promo_score + rng.normal(0, 0.9, n) >= 4).astype(int)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "income": income,
        "tenure_months": tenure_months,
        "past_purchases": past_purchases,
        "sessions_per_month": sessions_per_month,
        "avg_session_min": avg_session_min,
        "pages_per_session": pages_per_session,
        "cart_adds": cart_adds,
        "wishlist_items": wishlist_items,
        "days_since_last_visit": days_since_last_visit,
        "email_open_rate": email_open_rate,
        "used_coupon_before": used_coupon_before,
        "favorite_category": favorite_category,
        "will_purchase": will_purchase,
        "responded_to_promo": responded,
    })


def make_ratings(customers, products, n_customers=300):
    """Ratings 1-5 for ~25% of products per customer; customers prefer their
    favorite category, which gives KNN real structure to find."""
    rows = []
    for _, c in customers.head(n_customers).iterrows():
        rated = rng.choice(products.product_id, size=rng.integers(3, 8), replace=False)
        for pid in rated:
            cat = products.loc[products.product_id == pid, "category"].iloc[0]
            mean = 4.3 if cat == c.favorite_category else 2.6
            r = int(np.clip(round(rng.normal(mean, 0.8)), 1, 5))
            rows.append({"customer_id": int(c.customer_id), "product_id": int(pid), "rating": r})
    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    products = make_products()
    customers = make_customers()
    datasets = {
        "products": products,
        "sales": make_sales(products),
        "reviews": make_reviews(products),
        "customers": customers,
        "ratings": make_ratings(customers, products),
    }
    for name, df in datasets.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
