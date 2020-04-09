import pandas
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

df = pandas.read_csv('data-film.csv')
df1 = df[['AVGRATING_WEBSITE_1', 'AVGRATING_WEBSITE_3', 'GENRE']]

sns.scatterplot(x='AVGRATING_WEBSITE_1', y='AVGRATING_WEBSITE_3', data=df1, hue="GENRE", style="GENRE")


plt.show()
