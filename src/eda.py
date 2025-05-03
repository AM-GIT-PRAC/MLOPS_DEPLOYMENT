# -*- coding: utf-8 -*-

import pandas as pd           # To manage our Lego table (DataFrame)
import matplotlib.pyplot as plt  # To draw pictures (graphs)
import seaborn as sns         # For colorful pictures (charts)

# Step 1: Load the data
df = pd.read_csv('data/transactions.csv')

# Step 2: Get a first look at the data
print("��� First 5 transactions:")
print(df.head())  # Look at the first 5 bricks (transactions)

# Step 3: Check for missing values
print("\n��� Check for missing data:")
print(df.isnull().sum())  # Are any bricks missing?

# Step 4: Describe the data (see the size, min, max, mean)
print("\n��� Data Summary:")
print(df.describe())

# Step 5: Visualize the data with a histogram (show the distribution of transaction amounts)
plt.figure(figsize=(10, 6))
sns.histplot(df['transaction_amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts ���')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Step 6: Check for fraud (Look for weird bricks)
fraud_df = df[df['is_fraud'] == 1]
print("\n❗ Fraudulent Transactions:")
print(fraud_df.head())

# Step 7: Visualize fraud vs non-fraud transactions
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=df)
plt.title('Fraud vs Non-Fraud Transactions ���')
plt.xlabel('Fraud')
plt.ylabel('Count')
plt.show()

