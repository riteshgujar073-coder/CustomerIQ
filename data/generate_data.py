"""
Generate realistic e-commerce customer transaction and behavioral dataset
for the CustomerIQ platform.
"""

import numpy as np
import pandas as pd
import os

def generate_customer_dataset(num_customers: int = 2500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # Define archetypes with realistic parameters to ensure distinct, discoverable clusters
    archetypes = [
        # 1. VIP Champions (high frequency, high monetary, very recent, low discount)
        {
            "name": "Champions",
            "weight": 0.18,
            "recency": (5, 45, 12),           # min, max, mean
            "frequency": (15, 45, 24),
            "monetary": (3500, 12000, 6200),
            "visits": (60, 200, 110),
            "browse_time": (12, 35, 22),
            "discount_ratio": (0.02, 0.25, 0.10),
            "return_rate": (0.01, 0.12, 0.05),
            "tenure": (18, 60, 36),
            "satisfaction": (4, 5, 4.6),
            "categories": ["Electronics", "Fashion", "Beauty & Health"]
        },
        # 2. Loyal Steady Spenders (moderate-high frequency, good monetary, regular recency)
        {
            "name": "Loyal Steady",
            "weight": 0.28,
            "recency": (20, 90, 45),
            "frequency": (8, 20, 13),
            "monetary": (1200, 3600, 2100),
            "visits": (30, 90, 55),
            "browse_time": (8, 22, 14),
            "discount_ratio": (0.10, 0.45, 0.25),
            "return_rate": (0.02, 0.15, 0.07),
            "tenure": (12, 48, 26),
            "satisfaction": (3.5, 5, 4.2),
            "categories": ["Home & Kitchen", "Fashion", "Electronics", "Sports & Outdoors"]
        },
        # 3. Bargain & Discount Hunters (moderate frequency, lower monetary, high discount sensitivity)
        {
            "name": "Deal Seekers",
            "weight": 0.24,
            "recency": (15, 120, 50),
            "frequency": (5, 16, 9),
            "monetary": (400, 1400, 750),
            "visits": (40, 140, 75),
            "browse_time": (10, 28, 17),
            "discount_ratio": (0.45, 0.95, 0.70),
            "return_rate": (0.05, 0.25, 0.13),
            "tenure": (6, 36, 18),
            "satisfaction": (3.0, 4.5, 3.8),
            "categories": ["Fashion", "Beauty & Health", "Home & Kitchen"]
        },
        # 4. At-Risk / Churning High-Value (long recency, previously high frequency/monetary)
        {
            "name": "At-Risk",
            "weight": 0.16,
            "recency": (140, 360, 220),
            "frequency": (2, 10, 5),
            "monetary": (600, 3000, 1450),
            "visits": (5, 35, 16),
            "browse_time": (3, 14, 7),
            "discount_ratio": (0.05, 0.40, 0.20),
            "return_rate": (0.08, 0.32, 0.18),
            "tenure": (14, 52, 28),
            "satisfaction": (2.0, 3.8, 2.9),
            "categories": ["Electronics", "Home & Kitchen", "Sports & Outdoors"]
        },
        # 5. New / Promising Explorers (very recent, low frequency, short tenure)
        {
            "name": "New Explorers",
            "weight": 0.14,
            "recency": (2, 40, 18),
            "frequency": (1, 4, 2),
            "monetary": (80, 650, 260),
            "visits": (10, 60, 28),
            "browse_time": (6, 25, 13),
            "discount_ratio": (0.10, 0.60, 0.30),
            "return_rate": (0.01, 0.10, 0.04),
            "tenure": (1, 6, 3),
            "satisfaction": (3.5, 5, 4.1),
            "categories": ["Beauty & Health", "Fashion", "Electronics", "Sports & Outdoors"]
        }
    ]

    all_customers = []
    cust_id_counter = 10001
    
    for arch in archetypes:
        n_arch = int(num_customers * arch["weight"])
        
        # Sample realistic continuous variables with truncated normal distributions
        rec = np.clip(np.random.normal(arch["recency"][2], (arch["recency"][1] - arch["recency"][0]) / 5.5, n_arch), arch["recency"][0], arch["recency"][1]).round(0).astype(int)
        freq = np.clip(np.random.normal(arch["frequency"][2], (arch["frequency"][1] - arch["frequency"][0]) / 5.5, n_arch), arch["frequency"][0], arch["frequency"][1]).round(0).astype(int)
        mon = np.clip(np.random.normal(arch["monetary"][2], (arch["monetary"][1] - arch["monetary"][0]) / 5.5, n_arch), arch["monetary"][0], arch["monetary"][1]).round(2)
        visits = np.clip(np.random.normal(arch["visits"][2], (arch["visits"][1] - arch["visits"][0]) / 5.5, n_arch), arch["visits"][0], arch["visits"][1]).round(0).astype(int)
        browse = np.clip(np.random.normal(arch["browse_time"][2], (arch["browse_time"][1] - arch["browse_time"][0]) / 5.5, n_arch), arch["browse_time"][0], arch["browse_time"][1]).round(1)
        discount = np.clip(np.random.normal(arch["discount_ratio"][2], (arch["discount_ratio"][1] - arch["discount_ratio"][0]) / 5.5, n_arch), arch["discount_ratio"][0], arch["discount_ratio"][1]).round(2)
        returns = np.clip(np.random.normal(arch["return_rate"][2], (arch["return_rate"][1] - arch["return_rate"][0]) / 5.5, n_arch), arch["return_rate"][0], arch["return_rate"][1]).round(3)
        tenure = np.clip(np.random.normal(arch["tenure"][2], (arch["tenure"][1] - arch["tenure"][0]) / 5.5, n_arch), arch["tenure"][0], arch["tenure"][1]).round(0).astype(int)
        satisfaction = np.clip(np.random.normal(arch["satisfaction"][2], 0.45, n_arch), 1.0, 5.0).round(1)
        
        # Demographic attributes
        ages = np.clip(np.random.normal(36, 12, n_arch), 18, 72).round(0).astype(int)
        genders = np.random.choice(["Female", "Male", "Non-Binary"], size=n_arch, p=[0.52, 0.44, 0.04])
        categories = np.random.choice(arch["categories"], size=n_arch)
        
        # Derived values
        aov = (mon / freq).round(2)
        items_per_order = np.clip(np.random.normal(2.5, 0.8, n_arch), 1.0, 8.0)
        total_items = (freq * items_per_order).round(0).astype(int)
        
        for i in range(n_arch):
            all_customers.append({
                "CustomerID": f"CUST-{cust_id_counter}",
                "Age": ages[i],
                "Gender": genders[i],
                "Tenure_Months": tenure[i],
                "Recency_Days": rec[i],
                "Frequency": freq[i],
                "Monetary_Spend": mon[i],
                "Avg_Order_Value": aov[i],
                "Total_Items_Purchased": total_items[i],
                "Website_Visits": visits[i],
                "Avg_Session_Duration_Min": browse[i],
                "Discount_Usage_Ratio": discount[i],
                "Return_Rate": returns[i],
                "Satisfaction_Score": satisfaction[i],
                "Preferred_Category": categories[i]
            })
            cust_id_counter += 1

    df = pd.DataFrame(all_customers)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "customer_data.csv")
    df = generate_customer_dataset(num_customers=2500)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customer records saved to {output_path}")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
