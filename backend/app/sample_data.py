"""Generates a realistic sales / customers / transactions demo set with built-in patterns."""
import numpy as np
import pandas as pd


def make():
    rng = np.random.default_rng(7)
    prod = pd.DataFrame({
        "product_id": range(1, 11),
        "product_name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Phone", "Phone Case", "Charger",
                         "Headphones", "Desk", "Chair"],
        "category": ["Computing", "Accessories", "Accessories", "Computing", "Mobile", "Mobile", "Mobile",
                     "Audio", "Furniture", "Furniture"],
        "price": [900, 25, 60, 280, 650, 18, 30, 120, 220, 160]})
    n_c = 400
    cust = pd.DataFrame({
        "customer_id": range(1, n_c + 1), "age": rng.integers(18, 70, n_c).astype(float),
        "city": rng.choice(["Delhi", "Mumbai", "Chandigarh", "Pune", "Bengaluru"], n_c),
        "loyalty_tier": rng.choice(["Basic", "Silver", "Gold"], n_c, p=[0.6, 0.28, 0.12]),
        "signup_date": (pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, n_c), "D")).strftime("%Y-%m-%d")})
    cust.loc[rng.choice(n_c, 15, replace=False), "age"] = np.nan
    cust = pd.concat([cust, cust.sample(5, random_state=1)])  # a few duplicates to clean
    days = pd.date_range("2024-01-01", "2025-12-31")
    w = np.where(days.month.isin([11, 12]), 2.2, np.where(days.month.isin([6, 7]), 0.8, 1.0)) * (1 + 0.5 * np.arange(len(days)) / len(days))
    w = w / w.sum()
    pop = np.array([.14, .14, .08, .08, .14, .06, .06, .12, .09, .09])
    cust_w = np.where(cust.customer_id.values[:n_c] % 7 == 0, 4.0, 1.0)
    rows = []
    for oid in range(1, 1801):
        cid = int(rng.choice(np.arange(1, n_c + 1), p=cust_w / cust_w.sum()))
        day = rng.choice(days, p=w)
        first = int(rng.choice(np.arange(1, 11), p=pop))
        items = {first}
        if first == 1 and rng.random() < .6: items.add(2)
        if first == 5:
            if rng.random() < .55: items.add(6)
            if rng.random() < .4: items.add(7)
        if first == 4 and rng.random() < .35: items.add(3)
        if rng.random() < .25: items.add(int(rng.integers(1, 11)))
        mon = pd.Timestamp(day).month
        disc = float(rng.choice([0, 0, .05, .1, .2] if mon not in (11, 12) else [.1, .2, .2, .3]))
        ch = str(rng.choice(["web", "store", "app"], p=[.45, .35, .2]))
        for p in items:
            q = int(rng.integers(1, 4)) if prod.price[p - 1] < 100 else 1
            rows.append((oid, cid, p, q, disc, pd.Timestamp(day).strftime("%Y-%m-%d"), ch))
    tx = pd.DataFrame(rows, columns=["transaction_id", "customer_id", "product_id", "quantity", "discount", "date", "channel"])
    tx["revenue"] = (tx.product_id.map(prod.set_index("product_id").price) * tx.quantity * (1 - tx.discount)).round(2)
    tx.loc[rng.choice(len(tx), 40, replace=False), "channel"] = None
    return {"transactions": tx, "customers": cust, "products": prod}
