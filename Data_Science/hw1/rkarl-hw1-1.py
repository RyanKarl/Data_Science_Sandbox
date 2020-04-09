import pandas
df = pandas.read_csv('data-film.csv')
from scipy.stats import zscore

df1 = df[['AVGRATING_WEBSITE_1', 'AVGRATING_WEBSITE_2', 'AVGRATING_WEBSITE_3', 'AVGRATING_WEBSITE_4']]
df1 = df1.apply(zscore)
print("The maximum Z-Scores are: \n")
print(df1.max().to_string())
print("\nThe minimum Z-Scores are: \n")
print(df1.min().to_string())


