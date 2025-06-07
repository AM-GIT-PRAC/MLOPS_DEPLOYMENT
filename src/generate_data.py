# src/generate_data.py
# Static data generation - run once and commit to GitHub

import os
import pandas as pd
import random
import numpy as np
from faker import Faker
import json

fake = Faker()

# Define consistent categorical values
MERCHANTS = ['Amazon', 'Walmart', 'eBay', 'Target', 'BestBuy']
CARD_TYPES = ['Visa', 'MasterCard', 'Amex']
LOCATIONS = [
    'NewYork', 'LosAngeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'SanAntonio', 'SanDiego', 'Dallas', 'SanJose',
    'Austin', 'Jacksonville', 'SanFrancisco', 'Columbus', 'Charlotte'
]

def generate_static_dataset(num_samples=5000, fraud_ratio=0.3, seed=42):
    """
    Generate static dataset that will be committed to GitHub
    Larger dataset for better training
    """
    random.seed(seed)
    np.random.seed(seed)
    fake.seed_instance(seed)
    
    print(f"🎲 Generating static dataset: {num_samples} transactions with {fraud_ratio:.1%} fraud rate...")
    
    data = []
    
    for i in range(num_samples):
        # Determine if this transaction is fraud
        is_fraud = 1 if random.random() < fraud_ratio else 0
        
        # Generate features with realistic patterns
        if is_fraud:
            # Fraudulent transactions - higher amounts, specific patterns
            amount = round(random.uniform(200.0, 3000.0), 2)
            merchant = random.choices(MERCHANTS, weights=[0.4, 0.2, 0.2, 0.1, 0.1])[0]
            location = random.choice(LOCATIONS)
            card_type = random.choices(CARD_TYPES, weights=[0.5, 0.3, 0.2])[0]
        else:
            # Legitimate transactions - normal patterns
            amount = round(random.uniform(5.0, 800.0), 2)
            merchant = random.choice(MERCHANTS)
            location = random.choice(LOCATIONS)
            card_type = random.choice(CARD_TYPES)
        
        # Add transaction ID for tracking
        transaction_id = f"TXN_{i+1:06d}"
        
        data.append({
            'transaction_id': transaction_id,
            'amount': amount,
            'merchant': merchant,
            'location': location,
            'card_type': card_type,
            'is_fraud': is_fraud
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save main dataset
    output_file = 'data/transactions.csv'
    df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    fraud_count = df['is_fraud'].sum()
    legitimate_count = len(df) - fraud_count
    avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
    avg_legitimate_amount = df[df['is_fraud'] == 0]['amount'].mean()
    
    # Create comprehensive metadata
    metadata = {
        'dataset_info': {
            'total_transactions': len(df),
            'fraud_transactions': int(fraud_count),
            'legitimate_transactions': int(legitimate_count),
            'fraud_rate': float(fraud_count / len(df)),
            'avg_fraud_amount': float(avg_fraud_amount),
            'avg_legitimate_amount': float(avg_legitimate_amount),
            'generation_date': pd.Timestamp.now().isoformat(),
            'seed_used': seed
        },
        'features': {
            'merchants': MERCHANTS,
            'locations': LOCATIONS,
            'card_types': CARD_TYPES,
            'numerical_features': ['amount'],
            'categorical_features': ['merchant', 'location', 'card_type']
        },
        'statistics': {
            'amount_stats': df['amount'].describe().to_dict(),
            'merchant_distribution': df['merchant'].value_counts().to_dict(),
            'card_type_distribution': df['card_type'].value_counts().to_dict(),
            'fraud_by_merchant': df.groupby('merchant')['is_fraud'].mean().to_dict(),
            'fraud_by_card_type': df.groupby('card_type')['is_fraud'].mean().to_dict()
        }
    }
    
    # Save metadata
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create feature mapping for consistent encoding
    feature_columns = ['amount']  # Numerical feature
    
    for merchant in MERCHANTS:
        feature_columns.append(f'merchant_{merchant}')
    
    for location in LOCATIONS:
        feature_columns.append(f'location_{location}')
    
    for card_type in CARD_TYPES:
        feature_columns.append(f'card_type_{card_type}')
    
    feature_mapping = {
        'feature_columns': feature_columns,
        'categorical_mappings': {
            'merchant': MERCHANTS,
            'location': LOCATIONS,
            'card_type': CARD_TYPES
        },
        'total_features': len(feature_columns)
    }
    
    with open('data/feature_mapping.json', 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    
    print(f"✅ Static dataset created at {output_file}")
    print(f"📊 Dataset summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraudulent: {fraud_count:,} ({fraud_count/len(df):.1%})")
    print(f"   Legitimate: {legitimate_count:,} ({legitimate_count/len(df):.1%})")
    print(f"   Average fraud amount: ${avg_fraud_amount:.2f}")
    print(f"   Average legitimate amount: ${avg_legitimate_amount:.2f}")
    print(f"   Total features after encoding: {len(feature_columns)}")
    
    print(f"\n📋 Sample data:")
    print(df.head())
    
    print(f"\n📈 Fraud distribution by merchant:")
    print(df.groupby('merchant')['is_fraud'].agg(['count', 'mean']))
    
    return df

if __name__ == '__main__':
    # Generate static dataset
    print("🎯 Generating static dataset for MLOps pipeline...")
    df = generate_static_dataset(num_samples=5000, fraud_ratio=0.3)
    
    print("\n" + "="*50)
    print("📝 IMPORTANT: Commit this data to GitHub!")
    print("="*50)
    print("Run these commands:")
    print("git add data/")
    print("git commit -m 'Add static training dataset'")
    print("git push origin MLOPS_Change_2")
    print("="*50)
