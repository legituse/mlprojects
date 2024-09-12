# Import packages
import pandas as pd
from scipy.cluster import hierarchy
from sklearn import cluster
import matplotlib.pyplot as plt
import re
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import seaborn as sns
from sklearn.cluster import DBSCAN

# Clustering Problem

# Read the dataset 
initial_data = pd.read_csv("df_arabica_clean.csv")

# Data preprocessing
# Extract important attributes that will be used later
important_data = initial_data[["Variety", "Altitude", "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean Cup", "Sweetness", "Total Cup Points"]]

# Function to convert altitude ranges to their average
def convert_range_to_average(altitude):
    # Extract all numeric values using regex
    numeric_values = [int(val) for val in re.findall(r'\d+', str(altitude))]
    
    # Check the number of numeric values
    if len(numeric_values) == 1:
        # If there's only one number, return it as is
        return numeric_values[0]
    elif len(numeric_values) == 2:
        # If there are two numbers, return their mean
        return int(sum(numeric_values) / len(numeric_values))
    else:
        # If there are no numeric values or more than two, return None
        return None
    

# Apply the function to the 'Altitude' column
important_data['Altitude'] = important_data['Altitude'].apply(convert_range_to_average)

# Remove rows with null values
cleaned_data = important_data.dropna()

# Remove rows with 'unknown' in the 'Variety' column
cleaned_data = cleaned_data[~cleaned_data['Variety'].isin(['unknown', 'unknow'])]

# Extracting the Numerical Data Attributes
numerical_data = cleaned_data[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean Cup", "Sweetness"]]


# Print the counts of each variety
variety_counts = cleaned_data['Variety'].value_counts()
print("Counts of Variety in cleaned_data:")
for variety, count in variety_counts.items():
    print(f"{variety: <20} Count: {count}")

# K-Means Clustering On 9 Sensory Attributes 

# Creating an array to hold the various SSE's at different k-cluster sizes in kmeans
sse_list=[]

# Set max cluster range to 50, about 1/4 of the dataset size
# We can typically only have as many clusters as we have data points but 
# elbow point can be found within the first 50 (setting the limit also speeds up the code)
# random state is used for consistent results
cluster_range=list(range(1, 50)) 
for cluster_count in cluster_range:
    kMeans = cluster.KMeans(n_clusters=cluster_count, random_state=0)
    kMeans.fit(numerical_data)
    sse_list.append(kMeans.inertia_)

plt.figure()
plt.title("SSE vs Number of Clusters Graph")
plt.plot(cluster_range, sse_list)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE for Cluster')

# Looking at the K-Means graph we see at about k= 5 clusters is the best following the elbow method
k = 5

# using k = 5 clusters to run our k-means clustering algorithm
kMeans = cluster.KMeans(n_clusters=k,random_state=0)
kMeans.fit(numerical_data)
labels = kMeans.labels_

# Print the count and percentage of each variety in each cluster
clusters = pd.DataFrame(labels, columns=['Cluster ID'])
clusters['Variety'] = cleaned_data['Variety']

for cluster_id in range(k):
    cluster_varieties = clusters[clusters['Cluster ID'] == cluster_id]['Variety']
    variety_counts = cluster_varieties.value_counts()
    
    total_varieties_in_cluster = len(cluster_varieties)
    
    print(f"\nCluster {cluster_id + 1} Varieties:")
    for variety, count in variety_counts.items():
        percentage = (count / total_varieties_in_cluster) * 100
        print(f"\t{variety: <20} Count:{count: <4}, Percentage:{percentage:.2f}%")


# Create Scatter matrix following https://mlbhanuyerra.github.io/2018-02-19-Clustering-K-means/
scatter_matrix(numerical_data, c=labels, marker='o', cmap='Set1')

# Heirarchial Clustering
# Single link (MIN) dendrogram
min_analysis = hierarchy.single(numerical_data)
plt.figure() 
plt.title("Single Link Dendrogram for Coffee Quality Scores")
dn_min = hierarchy.dendrogram(min_analysis, orientation='right')


# Complete Link (MAX) dendrogram
max_analysis = hierarchy.complete(numerical_data)
plt.figure()
plt.title("Complete Link Dendrogram for Coffee Quality Scores")
dn_max = hierarchy.dendrogram(max_analysis, orientation='right')


altitude_total_points_data = cleaned_data[['Altitude', 'Total Cup Points']]
# Performing hierarchical clustering using single linkage on altitude and total cup points
min_analysis = hierarchy.single(altitude_total_points_data)
plt.figure()
plt.title("Single Link Dendrogram for Altitude and Total Cup Points")
dn_min = hierarchy.dendrogram(min_analysis, orientation='right')
plt.xlabel('Distance')
plt.ylabel('Coffee Id\'s')


## Density Based Clustering using DBSCAN
# Based on https://www.reneshbedre.com/blog/dbscan-python.html
# and https://www.section.io/engineering-education/dbscan-clustering-in-python/
# Did not work since dataset has a uniform density. Code is commented out
# but is kept here for reference and discussion in the report.

# # minPts should be 2 * number of dimensions
# minPts = 2 * len(numerical_data.columns)

# # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
# nbrs = NearestNeighbors(n_neighbors = (minPts+1)).fit(numerical_data)
# # Find the k-neighbors of a point
# neigh_dist, neigh_ind = nbrs.kneighbors(numerical_data)
# # sort the neighbor distances (lengths to points) in ascending order
# # axis = 0 represents sort along first axis i.e. sort along row
# sort_neigh_dist = np.sort(neigh_dist, axis = 0)

# k_dist = sort_neigh_dist[:, minPts]
# plt.plot(k_dist)
# plt.ylabel("k-NN distance")
# plt.xlabel(f'Sorted observations ({minPts}th NN)')
# plt.show()

# kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
#                       curve = "concave", direction = "increasing", online=True)

# # get the estimate of knee point
# kneedle.plot_knee()
# plt.show()

# EPS = kneedle.knee_y

# # Apply DBScan: eps set to one we calculated and minpts . 
# DBScanAnalysis = DBSCAN(eps=EPS, min_samples=minPts).fit(numerical_data)

# data_copy = numerical_data.copy()

# # Add DBSCAN labels to the data
# data_copy['Cluster'] = DBScanAnalysis.labels_
# # Remove all noise
# data_copy = data_copy[data_copy['Cluster'] != -1]

# # Plot pair plots
# plt.figure()
# sns.pairplot(data_copy, hue='Cluster', palette='viridis')


