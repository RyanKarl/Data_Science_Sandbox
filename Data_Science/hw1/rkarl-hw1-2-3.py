import pandas
from matplotlib import pyplot as plt

df = pandas.read_csv('data-film.csv')
from scipy.stats import zscore

#df1 = df[['AVGRATING_WEBSITE_1', 'AVGRATING_WEBSITE_2', 'AVGRATING_WEBSITE_3', 'AVGRATING_WEBSITE_4']]
#df1 = df1.apply(zscore)

#df.update(df1)
index_rom = index_act = index_com = score_rom= score_act = score_com = 0

for row in df.itertuples():
    if row[6] == 'ROMANCE':
        index_rom += 1
        score_rom += row[2]
    if row[6] == 'ACTION':
         index_act += 1
         score_act += row[2]
    if row[6] == 'COMEDY':
         index_com += 1
         score_com += row[2]

final_rom = score_rom/index_rom
final_com = score_com/index_com
final_act = score_act/index_act

bar_plot = pandas.DataFrame({'GENRE':['ROMANCE', 'ACTION', 'COMEDY'], 'RATING':[final_rom, final_act, final_com]})
ax = bar_plot.plot.bar(x='GENRE', y='RATING', rot=0)

plt.show()
