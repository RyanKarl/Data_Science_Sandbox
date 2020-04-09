## Q2 K Means Clustering ##
### Manhattan Distance Metric ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import HW3Functions as func
import math

### Initialisation
features = ["Rank_2015", "Rank_2017"]
colors = {1: 'r', 2: 'b'}
k = 2
### import data ###
d = pd.read_csv('hw3-data.txt', sep='\t')

### First Group of Centroids ###
## Given Initial Centroids ###
centroids1_0 = {1: [1,1],2:[25,25]}
d1_1=func.mdistMatrix(d, centroids1_0, features, colors, k)
#print d1_1

centroids1_1 = func.updateCentroids(d1_1, centroids1_0, features)
print(centroids1_1)
d1_2 = func.mdistMatrix(d1_1, centroids1_1, features, colors, k)
#print d1_2

centroids1_2 = func.updateCentroids(d1_2, centroids1_1, features)
print(centroids1_2)

### Repeat Iteration until updated centroids does not change ###
## cmp(dic1, dic2) is to compare whether two dictionary are same ##
## Here we use dic to record updated centroids
if centroids1_1 ==  centroids1_2:
    print("Centroids does not change. ")
else:
    print("Continue")

d1_3 = func.mdistMatrix(d1_2, centroids1_2, features, colors, k)
# print(d1_3)
centroids1_3 = func.updateCentroids(d1_3, centroids1_2, features)
print(centroids1_3)

if centroids1_2 ==  centroids1_3:
    print("Centroids does not change. ")
else:
    print("Continue")

### keep final data ###
finaldata1 = d1_3
cluster1_1=finaldata1[finaldata1['clusters'] == 1]['College'].tolist()
cluster1_2=finaldata1[finaldata1['clusters'] == 2]['College'].tolist()

### Print the results ###
print("Cluster1: " + str(cluster1_1))
print("Cluster2: " + str(cluster1_2))

print("SSE: " + str(func.SSE(finaldata1, k)))

### Plot the clusters ####
func.scatterplot('1st', finaldata1, features, colors, centroids1_3)

