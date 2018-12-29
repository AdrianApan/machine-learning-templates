#######################################################################################################################
"""
------------------
DATA PREPROCESSING
------------------
"""
#######################################################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

#####################################################################################
"""
------------------
K-MEANS CLUSTERING
------------------

• To be used to find categories/groups

• STEPS:
	(https://i.imgur.com/Brv5Gmw.png)
	1) Choose the number of K clusters
	2) Select at random K points, the centroids (not necessarily from your dataset)
	3) Assign each data point to the closest centroid -> That forms K clusters
	4) Compute and place the new centroid of each cluster
	5) Reassign each data point to the new closest centroid (basically step 3)
	   If any reassigment took place, go to step 4 otherwise FIN

• Visual explanation: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5714416?start=317

• Pay attention at random initialization trap:
  https://www.udemy.com/machinelearning/learn/v4/t/lecture/5714420?start=0

  To avoid this we need to use k-means++

"""
#####################################################################################


# Using the elbow method to find the optimal number of clusters
"""
• WCSS (Withing Clusters Sum of Square) formula: https://i.imgur.com/wOSx3UW.png

• WCSS explanation: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5714426?start=0

• WCSS is used to determine the right number of clusters.

• The Elbow method helps us choose the optimal number of clusters:
  https://i.imgur.com/IVY9Tir.png
"""
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # kmeans.inertia is the WCSS method from sklearn
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # Careful customers
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # Standard customers
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # Target customers
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') # Careless customers
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') # Sensible customers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()