import os
import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_transaction_data(num_samples=1000, fraud_ratio=0.3):
    data = []
    for _ in range(num_samples):
        is_fraud = 1 if random.random() < fraud_ratio else 0
        amount = round(random.uniform(1.0, 1000.0), 2)
        merchant = random.choice(['Amazon', 'Walmart', 'eBay', 'Target', 'BestBuy'])
        location = fake.city()
        card_type = random.choice(['Visa', 'MasterCard', 'Amex'])
        fraud_label = 'FRA' if is_fraud == 1 else 'NOR'
        data.append([amount, merchant, location, card_type, fraud_label])
    
    df = pd.DataFrame(data, columns=['amount', 'merchant', 'location', 'card_type', 'is_fraud'])

    # Create folder if not exists
    os.makedirs('data', exist_ok=True)
    
    df.to_csv('data/transactions.csv', index=False)
    print("✅ Dataset created at data/transactions.csv")

if __name__ == '__main__':
    generate_transaction_data()
