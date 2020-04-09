#### Apriori Algorithm ####
import numpy as np 
import pandas as pd
from itertools import combinations

### pandas ###
df = pd.read_csv('Q1.txt', sep = '\t')
df = df.loc[:, df.columns.values[1:5]]

### Apriori algorithm ####
# data: input data, here we use pandas Data Frame #
# minsup: minimum support #

def Apriori(data, minsup):
    # refine data by count freq of each item #
    refdata=pd.get_dummies(data.unstack().dropna()).groupby(level=1).sum()
    colnum, rownum  =refdata.shape
    pattern = []
    for cnt in range(1, rownum+1):
        for cols in combinations(refdata, cnt):
            support = refdata[list(cols)].all(axis=1).sum() # count support of each itemset
            pattern.append([",".join(cols), support])
    ## result data stores support of each itemset ##
    resdata = pd.DataFrame(pattern, columns=["Pattern", "Support"])
    results=resdata[resdata.Support >= minsup] # return the frequent patterns #

    return results

### print result given minimum support = 2 ###
print (Apriori(df,2))
