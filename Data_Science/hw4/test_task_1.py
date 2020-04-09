#http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

'''
Trans. ID Itemset
1 {a, b}
2 {b, c, d}
3 {a, c, d, e}
4 {a, d, e}
5 {a, b, c}
6 {a, b, c, d}
7 {a}
8 {a, b, c}
9 {a, b, d}
10 {b, c, e}
'''
'''
dataset = [['a', 'b'],
           ['b', 'c', 'd'],
           ['a', 'c', 'd', 'e'],
           ['a', 'd', 'e'],
           ['a', 'b', 'c'],
           ['a', 'b', 'c', 'd'],
           ['a'],
           ['a', 'b', 'c'],
           ['a', 'b', 'd'],
           ['b', 'c', 'e']]
'''

dataset = [['a', 'b'],
            ['b', 'c', 'd'],
            ['a', 'c', 'd', 'e'],
            ['a', 'd', 'e'],
            ['a', 'b', 'c'],
            ['a', 'b', 'c', 'd'],
            ['a'],
            ['a', 'b', 'c'],
            ['a', 'b', 'd'],
            ['b', 'c', 'e']]


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

from mlxtend.frequent_patterns import apriori

df = apriori(df, min_support=0.2, use_colnames=True)
print(df)

