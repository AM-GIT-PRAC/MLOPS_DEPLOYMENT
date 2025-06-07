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

# Generate a diverse set of locations
LOCATIONS = [
    'NewYork', 'LosAngeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'SanAntonio', 'SanDiego', 'Dallas', 'SanJose',
    'Austin', 'Jacksonville', 'SanFrancisco', 'Columbus', 'Charlotte',
    'Indianapolis', 'Seattle', 'Denver', 'Boston', 'Nashville'
]

def generate_transaction_data(num_samples=1000, fraud_ratio=0.3, seed=42):
    """
    Generate synthetic transaction data with realistic patterns
    
    Args:
        num_samples: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    fake.seed_instance(seed)
    
    print(f"🎲 Generating {num_samples} transactions with {fraud_ratio:.1%} fraud rate...")
    
    data = []
    
    for i in range(num_samples):
        # Determine if this transaction is fraud
        is_fraud = 1 if random.random() < fraud_ratio else 0
        
        # Generate features with realistic patterns
        if is_fraud:
            # Fraudulent transactions tend to have higher amounts and specific patterns
            amount = round(random.uniform(100.0, 2000.0), 2)
            merchant = random.choices(MERCHANTS, weights=[0.4, 0.2, 0.2, 0.1, 0.1])[0]  # Bias toward Amazon
            location = random.choice(LOCATIONS)
            card_type = random.choices(CARD_TYPES, weights=[0.5, 0.3, 0.2])[0]  # Bias toward Visa
        else:
            # Legitimate transactions have more normal patterns
            amount = round(random.uniform(1.0, 500.0), 2)
            merchant = random.choice(MERCHANTS)
            location = random.choice(LOCATIONS)
            card_type = random.choice(CARD_TYPES)
        
        # Add some noise and edge cases
        if random.random() < 0.05:  # 5% chance of unusual amounts
            amount = round(random.uniform(0.01, 50.0), 2)
        
        data.append({
            'amount': amount,
            'merchant': merchant,
            'location': location,
            'card_type': card_type,
            'is_fraud': is_fraud
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create folders if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Save data
    output_file = 'data/transactions.csv'
    df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    fraud_count = df['is_fraud'].sum()
    legitimate_count = len(df) - fraud_count
    avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
    avg_legitimate_amount = df[df['is_fraud'] == 0]['amount'].mean()
    
    # Create metadata
    metadata = {
        'dataset_info': {
            'total_transactions': len(df),
            'fraud_transactions': int(fraud_count),
            'legitimate_transactions': int(legitimate_count),
            'fraud_rate': float(fraud_count / len(df)),
            'avg_fraud_amount': float(avg_fraud_amount),
            'avg_legitimate_amount': float(avg_legitimate_amount)
        },
        'features': {
            'merchants': MERCHANTS,
            'locations': LOCATIONS,
            'card_types': CARD_TYPES
        },
        'generation_params': {
            'num_samples': num_samples,
            'fraud_ratio': fraud_ratio,
            'seed': seed
        }
    }
    
    # Save metadata
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset created at {output_file}")
    print(f"📊 Dataset summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraudulent: {fraud_count:,} ({fraud_count/len(df):.1%})")
    print(f"   Legitimate: {legitimate_count:,} ({legitimate_count/len(df):.1%})")
    print(f"   Average fraud amount: ${avg_fraud_amount:.2f}")
    print(f"   Average legitimate amount: ${avg_legitimate_amount:.2f}")
    print(f"📋 Features: {list(df.columns)}")
    print(f"🏷️ Categorical values:")
    print(f"   Merchants: {MERCHANTS}")
    print(f"   Card types: {CARD_TYPES}")
    print(f"   Locations: {len(LOCATIONS)} unique locations")
    
    return df

def create_feature_mapping():
    """Create a mapping of categorical features for consistent encoding"""
    
    # Generate all possible one-hot encoded column names
    feature_columns = ['amount']  # Numerical feature
    
    # Add merchant features
    for merchant in MERCHANTS:
        feature_columns.append(f'merchant_{merchant}')
    
    # Add location features
    for location in LOCATIONS:
        feature_columns.append(f'location_{location}')
    
    # Add card type features
    for card_type in CARD_TYPES:
        feature_columns.append(f'card_type_{card_type}')
    
    # Save feature mapping
    mapping = {
        'feature_columns': feature_columns,
        'categorical_mappings': {
            'merchant': MERCHANTS,
            'location': LOCATIONS,
            'card_type': CARD_TYPES
        }
    }
    
    with open('data/feature_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Feature mapping saved with {len(feature_columns)} total features")
    return feature_columns

if __name__ == '__main__':
    # Generate dataset
    df = generate_transaction_data()
    
    # Create feature mapping
    feature_columns = create_feature_mapping()
    
    # Show sample data
    print(f"\n📋 Sample data:")
    print(df.head())
    
    print(f"\n📈 Data types:")
    print(df.dtypes)
    
    print(f"\n📊 Value counts for categorical features:")
    for col in ['merchant', 'location', 'card_type']:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
