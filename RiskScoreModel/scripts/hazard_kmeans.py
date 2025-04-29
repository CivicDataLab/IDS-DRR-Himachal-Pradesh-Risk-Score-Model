import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import anderson


# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() + r"/HP/flood-data-ecosystem-Himachal-Pradesh"

# Load data
master_variables = pd.read_csv(path+'/RiskScoreModel/data/MASTER_VARIABLES.csv')
hazard_vars = ['inundation_intensity_mean_nonzero', 'inundation_intensity_sum', 'mean_rain', 'max_rain','drainage_density', 'Sum_Runoff', 'Peak_Runoff','slope_mean','elevation_mean','distance_from_river_mean']
hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]

hazard_df_months = []


# Define the KMeans clustering function for hazard scoring
def kmeans_scoring(df, n_clusters=5):
    # Select relevant features for clustering
    features = ['drainage_density', 'slope_mean','elevation_mean','distance_from_river_mean',
                'Sum_Runoff', 'Peak_Runoff', 'mean_rain', 'max_rain']
    
    # Normalize features to standardize their influence
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X)
    
    # Calculate mean hazard intensity for each cluster
    cluster_means = df.groupby('kmeans_cluster')['distance_from_river_mean'].mean()
    
    # Sort clusters by hazard intensity in descending order (highest hazard first)
    cluster_sorted = cluster_means.sort_values(ascending=False)
    
    # Create an inverted mapping from cluster label to hazard level
    inverse_levels = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}  # Mapping for inversion
    cluster_map = {cluster: inverse_levels[rank + 1] for rank, cluster in enumerate(cluster_sorted.index)}
    print("Inverted cluster mapping:", cluster_map)  # Debugging line
    
    # Map clusters to inverted hazard levels
    df['flood-hazard'] = df['kmeans_cluster'].map(cluster_map)
    
    # Final verification
    print("Assigned inverted hazard levels by cluster:")
    print(df[['kmeans_cluster', 'flood-hazard']].drop_duplicates().sort_values(by='kmeans_cluster'))
    
    return df['flood-hazard']

# Prepare monthly data
hazard_df_months = []

for month in tqdm(hazard_df.timeperiod.unique()):
    # Filter data for each month
    hazard_df_month = hazard_df[hazard_df.timeperiod == month]

    # Apply KMeans clustering for hazard scoring
    hazard_df_month['flood-hazard'] = kmeans_scoring(hazard_df_month)

    # Aggregate the results in a list
    hazard_df_months.append(hazard_df_month)

# Compile results
hazard = pd.concat(hazard_df_months)
master_variables = master_variables.merge(hazard[['timeperiod', 'object_id', 'flood-hazard']], on=['timeperiod', 'object_id'])

# Save the final results
master_variables.to_csv(path + r'/RiskScoreModel/data/hazard_k-means.csv', index=False)