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
-----------------------
HIERARCHICAL CLUSTERING
-----------------------

• There are 2 types of hierarchical clustering:
	1) Agglomerative (bottom-up approach)
	2) Divisive (reverse of agglomerative)

• Agglomerative HC steps:
	1) Make each data point a single-point cluster -> That forms N clusters
	2) Take the 2 closest data points and combine them to one cluster -> That forms N-1 clusters
	3) Take the 2 closest clusters and combine them to one cluster -> That forms N-2 clusters
	4) Repeat step 3 until there is 1 cluster left

	FOR STEP 3, DISTANCE BETWEEN CLUSTERS: https://i.imgur.com/zmf8Gup.png

"""
#####################################################################################

# Using the dendrogram to find the optimal number of clusters
"""

• How dendograms work: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5714432?start=0

• Using dendograms (set a dissimilarity threshold):
  https://www.udemy.com/machinelearning/learn/v4/t/lecture/5714438?start=0
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # Careful customers
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # Standard customers
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # Target customers
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') # Careless customers
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') # Sensible customers
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()