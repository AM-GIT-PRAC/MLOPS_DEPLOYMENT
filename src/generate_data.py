# -*- coding: utf-8 -*-

import pandas as pd      # Helps to create data tables like Lego trays ���
import numpy as np       # Helps to do math, like picking random stuff ���
import random            # Used for making random values (like surprise boxes!) ���

def generate_transactions(num_rows=1000):  # We're making 1000 fake transactions
    data = []  # This is our Lego box — we’ll put each transaction (brick) into it

    for _ in range(num_rows):  # Repeat 1000 times
        amount = round(random.uniform(1.0, 1000.0), 2)   # Lego brick: the amount of money
        time = random.randint(1, 86400)                 # Lego brick: time in seconds
        is_fraud = np.random.choice([0, 1], p=[0.97, 0.03])  # 3% chance it's fraud! ���

        # Build a transaction Lego brick:
        data.append({
            "transaction_time": time,
            "transaction_amount": amount,
            "location": random.choice(['USA', 'CAN', 'MEX', 'FRA', 'IND']),
            "card_type": random.choice(['VISA', 'MASTERCARD', 'AMEX']),
            "is_fraud": is_fraud
        })

    return pd.DataFrame(data)  # Make all those bricks into a table ���

# This will only run when we call the file directly
if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    print("✅ Fake transaction data generated!")

