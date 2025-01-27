import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split



def compute_similarity(features):
    """Calculate similarity matrix for feature set."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(scaled_features)
    return similarity_matrix

def build_customer_features(customers, transactions, products):
    """Generate comprehensive customer features from transactions."""
    # Aggregate transactional data
    transaction_summary = transactions.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'std', 'count'],
        'Quantity': ['sum', 'mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    transaction_summary.columns = ['CustomerID'] + ['_'.join(col).strip() for col in transaction_summary.columns[1:]]
    
    # Calculate recency (days since last transaction)
    latest_date = transactions['TransactionDate'].max()
    recency = transactions.groupby('CustomerID')['TransactionDate'].agg(
        lambda x: (latest_date - x.max()).days
    ).reset_index().rename(columns={'TransactionDate': 'Recency'})

    # Merge transaction summary and recency
    customer_data = pd.merge(transaction_summary, recency, on='CustomerID', how='left')
    
    # Map product categories to customer preferences
    merged_data = transactions.merge(products, on='ProductID', how='left')
    category_pref = merged_data.pivot_table(
        index='CustomerID',
        columns='Category',
        values='TotalValue',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Combine all features into one DataFrame
    final_features = pd.merge(customer_data, category_pref, on='CustomerID', how='left').fillna(0)
    return final_features


def get_similar_customers(customer_id, features, similarity_matrix, top_n=3):
    """Identify similar customers based on cosine similarity."""
    if customer_id not in features['CustomerID'].values:
        return []  # Handle invalid customer_id gracefully
    
    # Locate customer in feature matrix
    idx = features.index[features['CustomerID'] == customer_id].tolist()[0]
    similarity_scores = similarity_matrix[idx]
    
    # Sort indices of similar customers, excluding the given customer
    similar_indices = similarity_scores.argsort()[-(top_n + 1):-1][::-1]
    
    # Prepare results
    similar_customers = [
        {'CustomerID': features.iloc[i]['CustomerID'], 'SimilarityScore': round(similarity_scores[i], 4)}
        for i in similar_indices
    ]
    return similar_customers


  
def load_data():
    """Read datasets and ensure date columns are parsed."""
    customers = pd.read_csv('Customers.csv')
    products = pd.read_csv('Products.csv')
    transactions = pd.read_csv('Transactions.csv')

    # Convert dates to proper datetime format
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'], errors='coerce')
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')
    
    return customers, products, transactions

def train_similarity_model(features):
    """Train a model to predict customer categories."""
    X = features.drop(columns=['CustomerID'])
    y = (X['TotalValue_sum'] > X['TotalValue_sum'].mean()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model

def execute_pipeline():
    """Run the complete recommendation pipeline."""
    customers, products, transactions = load_data()
    customer_features = build_customer_features(customers, transactions, products)

    # Extract numerical data for similarity calculation
    numeric_features = customer_features.select_dtypes(include=[np.number]).copy()
    if 'CustomerID' in numeric_features.columns:
        numeric_features.drop(columns=['CustomerID'], inplace=True)

    # Compute similarity matrix
    similarity_matrix = compute_similarity(numeric_features)

    # Train the model
    train_similarity_model(customer_features)

    # Generate recommendations for the first 20 customers
    results = []
    for cust_id in customers['CustomerID'].iloc[:20]:
        recommendations = get_similar_customers(cust_id, customer_features, similarity_matrix, top_n=3)
        results.append({'CustomerID': cust_id, 'Recommendations': recommendations})

    # Save the results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('Swapnil_Adsul_Recommendations.csv', index=False)
    print("Recommendations saved to 'Swapnil_Adsul_Recommendations.csv'.")

if __name__ == "__main__":
    execute_pipeline()
