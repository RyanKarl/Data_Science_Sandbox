import pandas
from matplotlib import pyplot as plt

df = pandas.read_csv('data-film.csv')
from scipy.stats import zscore

df1 = df[['AVGRATING_WEBSITE_1', 'AVGRATING_WEBSITE_2', 'AVGRATING_WEBSITE_3', 'AVGRATING_WEBSITE_4']]
#df1 = df1.apply(zscore)

print df1.corr(method ='pearson')
