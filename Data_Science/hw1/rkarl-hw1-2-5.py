import pandas
from matplotlib import pyplot as plt
from scipy import stats
df = pandas.read_csv('data-film.csv')
from scipy.stats import zscore

bin1_1 = bin1_2 = bin1_3 = bin1_4 = bin1_5 = bin1_6 = bin1_7 = bin1_8 = 0
bin2_1 = bin2_2 = bin2_3 = bin2_4 = bin2_5 = bin2_6 = bin2_7 = bin2_8 = 0
total = 0

for row in df.itertuples():
     if row[2] >= 0 and row[2] <= 0.9:
         bin1_1 += 1
     if row[2] >= 1 and row[2] <= 1.9:
         bin1_2 += 1
     if row[2] >= 2 and row[2] <= 2.9:
         bin1_3 += 1
     if row[2] >= 3 and row[2] <= 3.9:
         bin1_4 += 1
     if row[2] >= 4 and row[2] <= 4.9:
         bin1_5 += 1
     if row[2] >= 5 and row[2] <= 5.9:
         bin1_6 += 1
     if row[2] >= 6 and row[2] <= 6.9:
         bin1_7 += 1
     if row[2] >= 7 and row[2] <= 7.9:
         bin1_8 += 1

     if row[4] >= 0 and row[4] <= 0.9:
         bin2_1 += 1
     if row[4] >= 1 and row[4] <= 1.9:
         bin2_2 += 1
     if row[4] >= 2 and row[4] <= 2.9:
         bin2_3 += 1
     if row[4] >= 3 and row[4] <= 3.9:
         bin2_4 += 1
     if row[4] >= 4 and row[4] <= 4.9:
         bin2_5 += 1
     if row[4] >= 5 and row[4] <= 5.9:
         bin2_6 += 1
     if row[4] >= 6 and row[4] <= 6.9:
         bin2_7 += 1
     if row[4] >= 7 and row[4] <= 7.9:
         bin2_8 += 1

     total += 1

ftotal = float(total)

dist1=[bin1_1/ftotal, bin1_2/ftotal, bin1_3/ftotal, bin1_4/ftotal, bin1_5/ftotal, bin1_6/ftotal, bin1_7/ftotal, bin1_8/ftotal]
dist3=[bin2_1/ftotal, bin2_2/ftotal, bin2_3/ftotal, bin2_4/ftotal, bin2_5/ftotal, bin2_6/ftotal, bin2_7/ftotal, bin2_8/ftotal]

print stats.entropy(pk=dist1, qk=dist3)

