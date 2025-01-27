import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def test_cluster_performance(features, max_clusters=10):
    """Evaluate clustering performance with different cluster counts."""
    performance_metrics = []
    for k in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        cluster_assignments = kmeans_model.fit_predict(features)
        
        davies_bouldin = davies_bouldin_score(features, cluster_assignments)
        silhouette_avg = silhouette_score(features, cluster_assignments)
        
        performance_metrics.append({
            'num_clusters': k,
            'davies_bouldin_index': davies_bouldin,
            'silhouette_score': silhouette_avg
        })
    
    return pd.DataFrame(performance_metrics)

def build_features(customers, transactions, products):
    """Construct features for clustering analysis."""
    # Calculate Recency, Frequency, and Monetary metrics
    max_transaction_date = transactions['TransactionDate'].max()
    rfm_data = transactions.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (max_transaction_date - x.max()).days,
        'TransactionID': 'count',
        'TotalValue': 'sum'
    }).rename(columns={
        'TransactionDate': 'Recency',
        'TransactionID': 'Frequency',
        'TotalValue': 'Monetary'
    })
    
    # Aggregate spending by product category
    category_spending = transactions.merge(products, on='ProductID')\
        .pivot_table(index='CustomerID', columns='Category', values='TotalValue', aggfunc='sum', fill_value=0)
    
    # Merge RFM metrics and category spending
    feature_set = pd.concat([rfm_data, category_spending], axis=1).fillna(0)
    
    return feature_set


def read_datasets():
    """Read input datasets and process date columns."""
    customer_info = pd.read_csv('Customers.csv')
    product_info = pd.read_csv('Products.csv')
    transaction_info = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customer_info['SignupDate'] = pd.to_datetime(customer_info['SignupDate'], errors='coerce')
    transaction_info['TransactionDate'] = pd.to_datetime(transaction_info['TransactionDate'], errors='coerce')
    
    return customer_info, product_info, transaction_info

def display_clusters(features, cluster_labels, save_as='cluster_plot.png'):
    """Visualize clusters using PCA reduction."""
    pca_reducer = PCA(n_components=2)
    reduced_features = pca_reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(save_as)
    plt.close()

def execute_clustering():
    # Load datasets
    customer_data, product_data, transaction_data = read_datasets()
    
    # Prepare features for clustering
    clustering_features = build_features(customer_data, transaction_data, product_data)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(clustering_features)
    
    # Apply KMeans clustering
    optimal_clusters = 7  # Assumed optimal
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_features)
    
    # Evaluate clustering performance
    silhouette_avg = silhouette_score(normalized_features, cluster_labels)
    davies_bouldin = davies_bouldin_score(normalized_features, cluster_labels)
    
    # Get sizes of clusters
    cluster_sizes = np.bincount(cluster_labels).tolist()
    
    # Save results to a text file
    results_file = 'Cluster_Results.txt'
    with open(results_file, 'w') as file:
        file.write("Clustering Results:\n")
        file.write(f"Cluster Sizes: {cluster_sizes}\n")
        file.write(f"Silhouette Score: {silhouette_avg}\n")
        file.write(f"Number of Clusters: {optimal_clusters}\n")
        file.write(f"Davies-Bouldin Index: {davies_bouldin}\n")
    
    print(f"Clustering completed. Results saved to '{results_file}'.")

if __name__ == "__main__":
    execute_clustering()
