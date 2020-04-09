#Ryan Karl

import pandas as pd
from matplotlib import pyplot as plt
import math
import seaborn as sns; sns.set()

#Function to calculate euclidean distance
def euclidean_distance(x, y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    return distance

#Function to calculate manhattan distance
def manhattan_distance(x,y):

    return sum(abs(a-b) for a,b in zip(x,y))

#Function to perform kmeans clustering
def kmeans_clustering(df, centroids):

    closest_centroid = []
    new_centroid_list1 = []
    new_centroid_list2 = []
    dist1_list = []
    dist2_list = []

    #Calculate the distance from each point to the centroids
    for row in df.itertuples():
        dist1 = (float(manhattan_distance([row[1], row[2]], centroids[0])))
        dist2 = (float(manhattan_distance([row[1], row[2]], centroids[1])))

        #Detrmine which centroid is closer
        if dist1 < dist2:
            closest_centroid.append(str(centroids[0]))
            new_centroid_list1.append([row[1], row[2]])
        else:
            closest_centroid.append(str(centroids[1]))
            new_centroid_list2.append([row[1], row[2]])

        dist1_list.append(dist1)
        dist2_list.append(dist2)

    num1 = 0
    num2 = 0

    #Collect centroid intermediate values for use calculating new centroids
    for val in new_centroid_list1:
        num1 += (val[0])
        num2 += (val[1])

    centroids[0][0] = float(num1)/len(new_centroid_list1)
    centroids[0][1] = float(num2)/len(new_centroid_list1)

    num1 = 0
    num2 = 0

    #Collect centroid intermediate values for use calculating new centroids
    for val in new_centroid_list2:
        num1 += (val[0])
        num2 += (val[1])

    centroids[1][0] = float(num1)/len(new_centroid_list2)
    centroids[1][1] = float(num2)/len(new_centroid_list2)

    df['dist1'] = dist1_list
    df['dist2'] = dist2_list
    df['closest_centroid'] = closest_centroid

    return df, centroids


#Read data and initialize centroids
df = pd.read_csv('Dataset-cluster.txt', sep='\t')
df = df[['Rank_2015', 'Rank_2017']]
centroids = [[1,1], [25,25]]

df, centroids = kmeans_clustering(df, centroids)

df['previous_closest_centroid'] = df['closest_centroid']

#Loop over kmeans function until the centroids don't change between iterations
while True:
    df, centroids = kmeans_clustering(df, centroids)

    if df['previous_closest_centroid'].equals(df['closest_centroid']):
        break

    else:
        df['previous_closest_centroid'] = df['closest_centroid']
        continue

print("Printing Plot with Starting Centroids (1,1) and (25,25)")
sns.scatterplot(x='Rank_2015', y='Rank_2017', data=df, palette=['blue','red'], hue="closest_centroid")

plt.show()
