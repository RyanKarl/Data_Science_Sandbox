## Functions ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

### Euclidean Distance Matrix ###
def edist(data, cen,features):
    x = data[features[0]].tolist()
    y = data[features[1]].tolist()
    D1 = []
    D2 = []
    for i in range(data.shape[0]):
        D1.append(round(math.sqrt(((x[i] - cen[1][0])**2 + (y[i] - cen[1][1])**2)),2))
        D2.append(round(math.sqrt(((x[i] - cen[2][0])**2 + (y[i] - cen[2][1])**2)),2))
    return D1, D2


### Manhattan Distance Matrix ###
def mdist(data, cen,features, k):
    x = data[features[0]].tolist()
    y = data[features[1]].tolist()
    if k == 2:
        D1 = []
        D2 = []
        for i in range(data.shape[0]):
            D1.append(round((abs(x[i] - cen[1][0]) + abs(y[i] - cen[1][1])),2))
            D2.append(round((abs(x[i] - cen[2][0]) + abs(y[i] - cen[2][1])),2))
        return D1, D2
    if k == 3:
        D1 = []
        D2 = []
        D3 = []
        for i in range(data.shape[0]):
            D1.append(round((abs(x[i] - cen[1][0]) + abs(y[i] - cen[1][1])),2))
            D2.append(round((abs(x[i] - cen[2][0]) + abs(y[i] - cen[2][1])),2))
            D3.append(round((abs(x[i] - cen[3][0]) + abs(y[i] - cen[3][1])),2))
        return D1, D2, D3

## Data Frame for Euclidean distance calculation and clusters assign
def edistMatrix(data, cen, features, colors):
    #print data
    #data1 = data
    for i in cen.keys():
        #print edist(data, cen)[i-1]
        data['distance_from_{}'.format(i)] = edist(data, cen, features)[i-1]
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in cen.keys()]

    data['clusters'] = data.loc[:, centroid_distance_cols].idxmin(axis=1)
    data['clusters'] = data['clusters'].map(lambda x: int(x.lstrip('distance_from_')))

    data['color'] = data['clusters'].map(lambda x: colors[x])
    return data

## Data Frame for Manhattan distance calculation and clusters assign
def mdistMatrix(data, cen, features, colors, k):
    for i in cen.keys():
        data['distance_from_{}'.format(i)] = mdist(data, cen, features, k)[i-1]
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in cen.keys()]
    data['clusters'] = data.loc[:, centroid_distance_cols].idxmin(axis=1)
    data['clusters'] = data['clusters'].map(lambda x: int(x.lstrip('distance_from_')))
    data['color'] = data['clusters'].map(lambda x: colors[x])
    return data

### Update Centroids
def updateCentroids(data, cen, features):
    cen1={}
    for i in cen.keys():
        cen1[i] = []
        cen1[i].append(np.mean(data[data['clusters'] == i][features[0]]))
        cen1[i].append(np.mean(data[data['clusters'] == i][features[1]]))
    return cen1

#### Scatter Plot of clusters #####
def scatterplot(f1, data, features, colors, centroids):
    fig = plt.figure(figsize=(5, 5))
    labels = data['College'].tolist()
    plt.scatter(data[features[0]], data[features[1]], color=data['color'], alpha=0.5)
    #for i in centroids1.keys():
    #    plt.scatter(*centroids1[i], color=colors[i])
    for label, x, y in zip(labels, data[features[0]], data[features[1]]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, 5),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
            )
    minx = min(data[features[0]])-1
    maxx = max(data[features[0]])+1
    miny = min(data[features[1]])-1
    maxy = max(data[features[1]])+1
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    #plt.title("Scatter Plot of Clusters with "+f1+" centroids initialisation")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    for key in centroids.keys():
        plt.scatter(centroids[key][0], centroids[key][1], c='black',  marker='^', alpha=1)
    plt.show()

### Evaluate which initialisation works better ###
def SSE(data1, k):

    L1 = []
    if k == 2:
        for i in range(data1.shape[0]):
            # print(min(data1['distance_from_1'][i], data1['distance_from_2'][i]))
            L1.append(min(data1['distance_from_1'][i], data1['distance_from_2'][i]))
    if k == 3:
        for i in range(data1.shape[0]):
            # print(min(data1['distance_from_1'][i], data1['distance_from_2'][i], data1['distance_from_3'][i]))
            L1.append(min(data1['distance_from_1'][i], data1['distance_from_2'][i], data1['distance_from_3'][i]))
    sse1 = round(sum(i*i for i in L1),2)
    return sse1

