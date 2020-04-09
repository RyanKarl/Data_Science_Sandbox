## Q1 K Means Clustering ##
### Euclidean Distance Metric ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import HW3Functions as func
import math

### Initialisation
features = ["Win_2015", "Win_2017"]
colors = {1: 'r', 2: 'b'}
k = 2
### import data ###
d = pd.read_csv('hw3-data.txt', sep='\t')

### First Group of Centroids ###
## Given Initial Centroids ###
centroids1_0 = {1: [7,7],2:[14,14]}
d1_1=func.edistMatrix(d, centroids1_0, features, colors)

centroids1_1 = func.updateCentroids(d1_1, centroids1_0, features)
print('{7, 7} and {14, 14} first time calculation:', centroids1_1)
d1_2 = func.edistMatrix(d1_1, centroids1_1, features, colors)

centroids1_2 = func.updateCentroids(d1_2, centroids1_1, features)
print('{7, 7} and {14, 14} second time calculation:', centroids1_2)


### Repeat Iteration until updated centroids does not change ###
## cmp(dic1, dic2) is to compare whether two dictionary are same ##
## Here we use dic to record updated centroids
if centroids1_1 == centroids1_2:
    print("Centroids does not change. ")
else:
    print("Continue")

### keep final data ###
finaldata1 = d1_2
cluster1_1=finaldata1[finaldata1['clusters'] == 1]['College'].tolist()
cluster1_2=finaldata1[finaldata1['clusters'] == 2]['College'].tolist()

### Print the results ###
print("Clusters of Group 1 centroids: ")
print("Cluster1: " + str(cluster1_1))
print("Cluster2: " + str(cluster1_2))


### Second Group of Centroids ###
## Given Initial Centroids ###
d = pd.read_csv('hw3-data.txt', sep='\t')
centroids2_0 = {1: [7,7],2:[7,14]}
d2_1=func.edistMatrix(d, centroids2_0, features, colors)

centroids2_1 = func.updateCentroids(d2_1, centroids2_0, features)
print('{7, 7} and {7, 14} first time calculation:', centroids2_1)
d2_2=func.edistMatrix(d2_1, centroids2_1, features, colors)

centroids2_2 = func.updateCentroids(d2_2, centroids2_1, features)
print('{7, 7} and {7, 14} second time calculation:', centroids2_2)

### Repeat Iteration until updated centroids does not change ###
## cmp(dic1, dic2) is to compare whether two dictionary are same ##
## Here we use dic to record updated centroids
if centroids2_1 == centroids2_2:
    print("Centroids does not change. ")
else:
    print("Continue")

### keep final data ###
finaldata2 = d2_2
cluster2_1=finaldata2[finaldata2['clusters'] == 1]['College'].tolist()
cluster2_2=finaldata2[finaldata2['clusters'] == 2]['College'].tolist()

### Print the results ###
print("Clusters of Group 2 centroids: ")
print("Cluster1: " + str(cluster2_1))
print("Cluster2: " + str(cluster2_2))

### Evaluate which initialisation works better ###
print("SSE of Group1: "+ str(func.SSE(finaldata1, k)))
print("SSE of Group2: "+ str(func.SSE(finaldata2, k)))

### Plot the clusters ####
func.scatterplot('1st', finaldata1, features, colors, centroids1_2)
func.scatterplot('2nd', finaldata2, features, colors, centroids2_2)
