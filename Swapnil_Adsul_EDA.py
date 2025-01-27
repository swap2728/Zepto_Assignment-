import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np



def explore_data(dataframe, label):
    """Perform exploratory data analysis for a dataset"""
    print(f"\nExploration of {label} Data:")
    print("-" * 50)
    print(f"Dataset Dimensions: {dataframe.shape}")
    print("\nSample Rows:")
    print(dataframe.isnull().sum())
    print("\nStatistical Summary:")
    print(dataframe.describe())
    print(dataframe.info())
    print("\nCount of Missing Values:")
    print(dataframe.head())
    print("\nColumn Information:")

def analyze_transactions(transactions, products):
    """Investigate transaction trends"""
    # Summarize by month
    monthly_trends = transactions.groupby(
        transactions['TransactionDate'].dt.to_period('M')
    )['TotalValue'].sum()
    
    # Visualize the trends
    plt.figure(figsize=(12, 6))
    monthly_trends.plot(kind='line')
    plt.title('Monthly Revenue Trends')
    plt.ylabel('Revenue')
    plt.xlabel('Month')
    plt.savefig('revenue_trend.png')
    plt.close()

def analyze_customers(customer_data, transaction_data):
    """Examine customer behavior metrics"""
    # Aggregate transaction metrics for customers
    customer_metrics = transaction_data.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    customer_metrics.columns = ['CustomerID', 'TransactionCount', 'Revenue', 'TotalItems']
    return customer_metrics

def load_files():
    """Load the datasets into DataFrames"""
    products = pd.read_csv('Swapnil_Adsul_Data/Products.csv')
    transactions = pd.read_csv('Swapnil_Adsul_Data/Transactions.csv')
    customers = pd.read_csv('Swapnil_Adsul_Data/Customers.csv')
    
    # Parse date columns
    
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
  
    return transactions,customers, products, 


def process_data():
    # Load and process data
    transactions,customers, products = load_files()
    
    # Conduct basic exploration
    explore_data(products, "Product Details")
    explore_data(transactions, "Transaction Records")
    explore_data(customers, "Customer Information")
    
    
    # Perform advanced analysis
    
    customer_statistics = analyze_customers(customers, transactions)
    analyze_transactions(transactions, products)
    
    # Display aggregated customer data
    print("\nCustomer Metrics:")
    print(customer_statistics.head())

if __name__ == "__main__":
    process_data()
