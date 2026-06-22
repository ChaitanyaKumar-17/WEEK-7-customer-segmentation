# Customer Segmentation System

A robust Python machine learning project designed to ingest behavioral and demographic data, preprocess features, and build a customer segmentation model using K-Means and Hierarchical clustering. This project emphasizes industry best practices in unsupervised learning, dimensional reduction, and extracting actionable business intelligence from unlabeled data.

## 💡 Overview

Modern businesses rely on clustering to identify distinct customer groups for targeted marketing, dynamic pricing strategies, and personalized product recommendations. This project automates a workflow for extracting insights from unlabeled datasets—one of the most valuable skills in AI-driven business intelligence. By scaling features, calculating optimal cluster counts, and visualizing the data via Principal Component Analysis (PCA), the model transforms raw demographic data into meaningful, targetable consumer profiles.

## ✨ Features

* **Flexible Data Acquisition:** Designed to ingest standard retail or e-commerce datasets (e.g., Mall Customers, Online Retail) containing behavioral and demographic attributes.
* **Distance-Bias Prevention:** Utilizes `scikit-learn`'s `StandardScaler` or `MinMaxScaler` to normalize disparate feature ranges (e.g., Age vs. Annual Income) so the clustering algorithm does not mathematically favor larger numerical values.
* **Algorithmic Optimization:** Dynamically determines the optimal number of clusters by mathematically evaluating the data using the Elbow Method and Silhouette Scores.
* **Multi-Algorithm Support:** Implements standard K-Means clustering while supporting optional Hierarchical Clustering to compare internal structure and cluster linkage.
* **Actionable Business Intelligence:** Analyzes the final mathematical groupings to apply meaningful, real-world labels (e.g., "Budget Shoppers," "Premium Buyers," "Occasional Visitors").

## 🛠️ Prerequisites

* Python 3.8 or higher
* A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
* Core Scientific Libraries: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

## 🚀 Usage

1. Clone this repository to your local machine.
2. Open your terminal or command prompt and install the necessary dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```
3. Launch your Python environment or execute the script directly:
   ```bash
   python app.py
   ```

## 📊 Expected Output

Upon successful execution, the script will process the raw customer data in memory and output both statistical metrics and visual insights, including:

1. **Optimal Cluster Diagnostics:** Visual outputs displaying the Elbow curve and Silhouette scores to mathematically justify the chosen number of clusters ('K').
2. **Dimensionality Reduction Visuals:** 2D scatter plots and pair plots generated via Principal Component Analysis (PCA), mapping complex multi-dimensional clusters into an easily readable visual format.
3. **Segment Profiles:** A structured breakdown comparing the aggregated demographics and spending patterns across the distinctly labeled customer clusters.

## 🧩 How It Works

This script serves as a practical application of standard unsupervised machine learning pipelines tailored for clustering algorithms:

1. **Data Ingestion & Cleaning:** The script loads the chosen dataset and immediately addresses any missing or inconsistent values to ensure strict data integrity.

2. **Standardization:** Continuous demographic features (like Age, Annual Income, and Spending Score) are routed through a scaler. This brings all features to a level playing field, preventing distance-based algorithms from skewing toward features with wider numerical ranges.

3. **Optimal 'K' Selection:** Before finalizing the clusters, the algorithm runs iterations to plot the Elbow Method and calculate Silhouette Scores, allowing the user to select the most mathematically sound number of groups.

4. **K-Means Clustering:** The normalized data flows into a K-Means algorithm, grouping customers based on their proximity to iteratively optimized centroids. (Optionally, Hierarchical Clustering builds a dendrogram to evaluate the data's nested structure).

5. **PCA Dimensionality Reduction:** To visualize clusters that contain more than two or three features, Principal Component Analysis (PCA) compresses the dataset into a 2D plane while retaining the maximum variance of the original data.

6. **Semantic Labeling:** The algorithm aggregates the data within each mathematical cluster, allowing the user to identify distinct behavioral patterns and assign actionable marketing labels to each group.