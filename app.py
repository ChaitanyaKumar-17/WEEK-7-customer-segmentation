import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("--- Step 1: Loading Dataset ---")
np.random.seed(42)
data = {
    'CustomerID': range(1, 201),
    'Age': np.random.randint(18, 70, 200),
    'Annual Income (k$)': np.random.randint(15, 140, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)
print(f"Dataset loaded with {df.shape[0]} customers.\n")

print("--- Step 2: Cleaning & Scaling ---")
# Drop irrelevant features for clustering (like CustomerID)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Handle missing values (if any)
X = X.dropna()

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features successfully scaled using StandardScaler.\n")