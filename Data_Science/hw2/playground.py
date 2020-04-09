import pandas
import math
df = pandas.read_csv('Dataset-football-train.txt', sep='\t')#, lineterminator='\r')
import numpy as np
import matplotlib.pyplot as plt

df = df.drop(columns=['ID', 'Date', 'Opponent'])

def IG(h, win, lose):
    h = float(h)
    win = float(win)
    lose = float(lose)
    if h == 0:
        return 0
    else:
        return(-(((win)/(win+lose)*(math.log((win/(win+lose)), 2)))+((lose)/(win+lose)*(math.log((lose/(win+  lose)), 2)))) - h)

def H_single(a, b, a_win, a_lose):
    a = float(a)
    b = float(b)
    a_win = float(a_win)
    a_lose = float(a_lose)

    if a_win == 0 and a_lose == 0:
        return 0
    elif a_win == 0:
        return (((a/(a+b))*(-((a_lose/a)*math.log((a_lose/a), 2)))))
    elif a_lose == 0:
        return (((a/(a+b))*(-((a_win/a)*math.log((a_win/a), 2)))))
    else:
        return (((a/(a+b))*(-(((a_win/a)*math.log((a_win/a), 2))+((a_lose/a)*math.log((a_lose/a), 2))))))


def Tree_Split(data, search_string):
    val_index = -1
    new_data = []
    for val in data:
        val_index += 1
        if search_string in val:dddddddiii
            #print(val)
            new_data.append(val)
            #new_data = np.delete(new_data, val_index, 0)
    for item in new_data:
        index = np.argwhere(item==search_string)
        item = np.delete(item, index)

        #print(item)
    #print(data)
    #print(" ")
    new_data = np.asarray(new_data)
    #print(new_data)
    #print data


'''
global COLUMN_HEADERS
COLUMN_HEADERS = df.columns
data = df.values

potential_splits = {}
_, n_columns = data.shape


for column_index in range(n_columns - 1):        # excluding the last column which is the label
    values = data[:, column_index]
    print(values)
    unique_values = np.unique(values)
    print(unique_values)
'''

global COLUMN_HEADERS
COLUMN_HEADERS = df.columns
data = df.values

_, n_columns = data.shape

home_win_count = home_lose_count = away_win_count = away_lose_count = away_count = home_count = in_win_count =        in_lose_count = out_win_count = out_lose_count = in_count = out_count = win_count = lose_count = NBC_count =          ABC_count = ESPN_count = FOX_count = CBS_count = ABC_win_count = ABC_lose_count = NBC_win_count = NBC_lose_count =    CBS_win_count = CBS_lose_count = FOX_win_count = FOX_lose_count = ESPN_win_count = ESPN_lose_count = H_In_Or_Out =    H_Home_Or_Away = H_Media = H_NBC = H_ABC = H_FOX = H_CBS = H_ESPN = IG_Media = IG_In_Or_Out = IG_Home_Or_Away = 0



for column_index in range(n_columns - 1):        # excluding the last column which is the label
    values = data[:, column_index]
    label_index = -1
    #This prints win or lose
    #print(data[:, n_columns-1])

    for item in values:
        #print(item)
        #if label_index <= 26:
        label_index += 1
        #else:
        #    label_index = 0

        #print(data[label_index, n_columns-1])
        #label_index += 1
        #print(win_count)
        if item == "Home" and data[label_index, n_columns-1] == "Win":
            home_win_count += 1
            home_count += 1
            win_count += 1

        elif item == "Home" and data[label_index, n_columns-1] == "Lose":
            home_lose_count += 1
            home_count += 1
            lose_count += 1

        elif item == "Away" and data[label_index, n_columns-1] == "Win":
            away_win_count += 1
            away_count += 1
            win_count += 1

        elif item == "Away" and data[label_index, n_columns-1] == "Lose":
            away_lose_count += 1
            away_count += 1
            lose_count += 1



#if item == "Home" and data[label_index, n_columns-1] == "Win"
        if item == "In" and data[label_index, n_columns-1] == "Win":
            in_win_count += 1
            in_count += 1
            win_count += 1
        elif item == "In" and data[label_index, n_columns-1] == "Lose":
            in_lose_count += 1
            in_count += 1
            lose_count += 1
        elif item == "Out" and data[label_index, n_columns-1] == "Win":
            out_win_count += 1
            out_count += 1
            win_count += 1
        elif item == "Out" and data[label_index, n_columns-1] == "Lose":
            out_lose_count += 1
            out_count += 1
            lose_count += 1

        if item == "NBC":
            if data[label_index, n_columns-1] == 'Win':
                NBC_win_count += 1
                NBC_count += 1
                win_count += 1
            elif data[label_index, n_columns-1] == 'Lose':
                NBC_lose_count += 1
                NBC_count += 1
                lose_count += 1
        elif item == "ABC":
            if data[label_index, n_columns-1] == 'Win':
                ABC_win_count += 1
                ABC_count += 1
                win_count += 1
            elif data[label_index, n_columns-1] == 'Lose':
                ABC_lose_count += 1
                ABC_count += 1
                lose_count += 1
        elif item == 'ESPN':
            if data[label_index, n_columns-1] == 'Win':
                ESPN_win_count += 1
                ESPN_count += 1
                win_count += 1
            elif data[label_index, n_columns-1] == 'Lose':
                ESPN_lose_count += 1
                ESPN_count += 1
                lose_count += 1
        elif item == 'FOX':
            if data[label_index, n_columns-1] == 'Win':
                FOX_win_count += 1
                FOX_count += 1
                win_count += 1
            elif data[label_index, n_columns-1] == 'Lose':
                FOX_lose_count += 1
                FOX_count += 1
                lose_count += 1
        elif item == 'CBS':
            if data[label_index, n_columns-1] == 'Win':
                CBS_win_count += 1
                CBS_count += 1
                win_count += 1
            elif data[label_index, n_columns-1] == 'Lose':
                CBS_lose_count += 1
                CBS_count += 1
                lose_count += 1

    #label_index += 1

for values in data:
    if 'Home' in values or 'Away' in values:

        H_Home_Or_Away = H_single(home_count, away_count, home_win_count, home_lose_count) + H_single(away_count, home_count, away_win_count, away_lose_count)
        IG_Home_Or_Away = IG(H_Home_Or_Away, win_count, lose_count)

    if 'In' in values or 'Out' in values:

        H_In_Or_Out = H_single(in_count, out_count, in_win_count, in_lose_count) + H_single(out_count, in_count, out_win_count, out_lose_count)
        IG_In_Or_Out = IG(H_In_Or_Out, win_count, lose_count)

    if 'ABC' in values or 'NBC' in values or 'CBS' in values or 'ESPN' in values or 'FOX' in values:
         H_NBC = H_single(NBC_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count), NBC_win_count, NBC_lose_count)
         H_ABC = H_single(ABC_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    ABC_win_count, ABC_lose_count)
         H_FOX = H_single(FOX_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    FOX_win_count, FOX_lose_count)
         H_CBS = H_single(CBS_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    CBS_win_count, CBS_lose_count)
         H_ESPN = H_single(ESPN_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    ESPN_win_count, ESPN_lose_count)

         H_Media = H_NBC + H_ABC + H_FOX + H_CBS + H_ESPN
         IG_Media = IG(H_Media, win_count, lose_count)

d = dict()
d['H_Home_Or_Away'] = float(H_Home_Or_Away)
d['IG_Home_Or_Away'] = float(IG_Home_Or_Away)
d['H_In_Or_Out'] = float(H_In_Or_Out)
d['IG_In_Or_Out'] = float(IG_In_Or_Out)
d['H_Media'] = float(H_Media)
d['IG_Media'] = float(IG_Media)

#print(d)
IG_Media = float(IG_Media)
IG_In_Or_Out = float(IG_In_Or_Out)
IG_Home_Or_Away = float(IG_Home_Or_Away)

     #if IG_Media == IG_In_Or_Out == IG_Home_Or_Away == 0:
     #    print("Reached End Leaf")
     #    return 0

if max(IG_Media, IG_In_Or_Out, IG_Home_Or_Away) == IG_Media:
    data1 = data.copy()
    data2 = data.copy()
    data3 = data.copy()
    data4 = data.copy()
    data5 = data.copy()
    data1 = Tree_Split(data1, "NBC")
    data2 = Tree_Split(data2, "ABC")
    data3 = Tree_Split(data3, "CBS")
    data4 = Tree_Split(data4, "FOX")
    data5 = Tree_Split(data5, "ESPN")

feature_name = "Is_In_Top_25"
split_value = ["NBC", "ABC", "CBS", "FOX", "ESPN"]
for x in split_value:
    question = "{} <= {}".format(feature_name, x)
    sub_tree = {question: []}
    print(sub_tree)

        #return [df1, df2, df3, df4, df5]
'''
    elif max(IG_Media, IG_In_Or_Out, IG_Home_Or_Away) == IG_In_Or_Out:
        df6 = dataframe.copy()
        df7 = dataframe.copy()
        df6 = Tree_Split(df6, "Is_Opponent_in_AP25_Preseason", "In")
        df7 = Tree_Split(df7, "Is_Opponent_in_AP25_Preseason", "Out")
        #return [df6, df7]

    else:
        df8 = dataframe.copy()
        df9 = dataframe.copy()
        df8 = Tree_Split(df8, "Is_Home_or_Away", "Home")
        df9 = Tree_Split(df9, "Is_Home_or_Away", "Away")

        #return [df8, df9]
'''


