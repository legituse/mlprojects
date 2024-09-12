# Import the packages
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN
from sklearn import cluster
from apyori import apriori
import matplotlib.pyplot as plt
import os



## Association Analysis ##

# Apiori #

# Read the dataset 
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "car.data"), header=None)

# Setting the column names for the car dataset
data.columns = ["buy","maint","doors","persons","lug_boot","safety","acceptance"]


# maping almost all categorical columns to it's own string value
data['buy'] = data['buy'].apply(lambda x: "buy_"+str(x))
data['maint'] = data['maint'].apply(lambda x: "maint_"+str(x))
data['doors'] = data['doors'].apply(lambda x: str(x)+"_doors")
data['persons'] = data['persons'].apply(lambda x: str(x)+"_person")
data['lug_boot'] = data['lug_boot'].apply(lambda x: "lug_boot_"+str(x))
data['safety'] = data['safety'].apply(lambda x: "safety_"+str(x))


# data = data[["name", "mainhue", "topleft", "botright"]]

# Making big list
biglist= []
for i in range(data.shape[0]):
    biglist.append(data.iloc[i].tolist())

# Apriori rules
rules = apriori(biglist, min_support = 0.2, min_confidence = 0.55)

# Print rules
print("Printing the Apriori Rules:")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support: {}%'.format(round(rule.support*100,3)), 'Confidence: {}%'.format(round(rule.ordered_statistics[0].confidence*100,3)))



# Importing the top 50 songs in 2021 Data set
initial_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "spotify_top50_2021.csv"))

# Extracting the Numerical Data
numerical_data = initial_data.iloc[:,4:]


## K-Means Clustering ##

# Creating an array to hold the various SSE's at different k-cluster sizes in kmeans
sse_list=[]

cluster_range=list(range(1,50)) # We can only have as many clusters as data points
for cluster_count in cluster_range:
    kMeans = cluster.KMeans(n_clusters=cluster_count)
    kMeans.fit(numerical_data)
    sse_list.append(kMeans.inertia_)

plt.figure()
plt.title("SSE vs Clusters for K-Means Graph")
plt.plot(cluster_range, sse_list)
plt.xlabel('# of Clusters')
plt.ylabel('SSE for Cluster')

# Looking at the K-Means graph we see k=5 clusters is the best following the elbow method
k = 5

# using k = 5 clusters to run our k-means clustering algorithm
kMeans = cluster.KMeans(n_clusters=k,random_state=0)
kMeans.fit(numerical_data)
labels = kMeans.labels_

# Printing the song name vs the cluster it is a part of
clusters = pd.DataFrame(labels, index=initial_data.track_name, columns=['Cluster ID'])
print("\nPrinting the K-Means Cluster for each Track Name\n",clusters)

## Heirarchial Clustering ##

# Single link (MIN) analysis + plot associated dendrogram ###
min_analysis = hierarchy.single(numerical_data)

plt.figure() 
plt.title("Single Link Dendrogram")
dn_min = hierarchy.dendrogram(min_analysis, labels = initial_data['track_name'].to_list(), orientation='right')


# Complete Link (MAX) analysis + plot associated dendrogram ###
max_analysis = hierarchy.complete(numerical_data)

plt.figure()
plt.title("Complete Link Dendrogram")
dn_max = hierarchy.dendrogram(max_analysis, labels = initial_data['track_name'].to_list(), orientation='right')


# Group Average analysis + plot associated dendrogram ###
average_analysis = hierarchy.average(numerical_data)

plt.figure()
plt.title("Group Average Dendrogram")
dn_average = hierarchy.dendrogram(average_analysis, labels = initial_data['track_name'].to_list(), orientation='right')



## Density Based Clustering: DBSCAN  ##

# Reading only the popularity and the duration of songs
data_points= initial_data[["popularity", "duration_ms"]]

# Determining the EPS and minpoint using Nearest Neighbors on each point and graphing it
# Referenced from https://www.section.io/engineering-education/dbscan-clustering-in-python/
clf = NearestNeighbors(n_neighbors=4) # Creating a classifier for 4 nearest neighbors
neighbours=clf.fit(data_points) # Fitting popularity vs duration into nearest neighbors
distances,indices=neighbours.kneighbors(data_points) # Extracting the distances between each points between the neighbors
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances


# Plotting the sorted distance of every point to its 4th nearest neighbor
plt.figure()
plt.plot(distances) # plotting the distances
plt.title("Nearest 4 Neighbours Graph to determine EPS")
plt.xlabel("Points Sorted According to Distance of 4th Nearest Neighbor")
plt.ylabel("4th Nearest Neighbor Distance")


# After looking at the graph we pick EPS = 8000 to be our elbow point 
EPS=8000


# Generating the Initial scatter plot for popularity vs song duration
plt.figure()
plt.scatter(data_points["popularity"], data_points["duration_ms"])
plt.title("Initial Scatter plot for Popularity vs Duration of Songs")
plt.xlabel("Popularity")
plt.ylabel("Duration (ms)")


# Apply DBScan: eps set to 5000 and minpts set to 4. 
DBScanAnalysis = DBSCAN(eps=EPS, min_samples=4).fit(data_points)
# Concatenate data with cluster labels:
# Convert labels as a pandas dataframe
clustersLabels = pd.DataFrame(DBScanAnalysis.labels_,columns=['Cluster ID'])
# Concatenate the dataframes 'data' and 'clustersLabels' 
result = pd.concat((data_points, pd.DataFrame(clustersLabels, columns=['Cluster ID'])), axis=1)

# scatter plot of the results data
# each point with coordinates x and y is represented as a dot; 
plt.figure()
plt.scatter(result["popularity"], result["duration_ms"], c=result['Cluster ID'], cmap='jet')
plt.title("Result Scatter plot for Popularity vs Duration of Songs")
plt.xlabel("Popularity")
plt.ylabel("Duration (ms)")



