import pandas
import math
df = pandas.read_csv('Dataset-football-train.txt', sep='\t')
import numpy as np
import matplotlib.pyplot as plt
from anytree import Node, RenderTree

def Tree_Split(dataframe, row_label, search_string):
    for index, row in dataframe.iterrows():
        if row[str(row_label)] != str(search_string):
            dataframe.drop(index, inplace=True)
    dataframe = dataframe.drop(columns=[row_label])

    return dataframe

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


def IG(h, win, lose):
    h = float(h)
    win = float(win)
    lose = float(lose)
    if h == 0:
        return 0
    else:
        return(-(((win)/(win+lose)*(math.log((win/(win+lose)), 2)))+((lose)/(win+lose)*(math.log((lose/(win+lose)), 2)))) - h)

home_win_count = home_lose_count = away_win_count = away_lose_count = away_count = home_count = in_win_count = in_lose_count = out_win_count = out_lose_count = in_count = out_count = win_count = lose_count = NBC_count = ABC_count = ESPN_count = FOX_count = CBS_count = ABC_win_count = ABC_lose_count = NBC_win_count = NBC_lose_count = CBS_win_count = CBS_lose_count = FOX_win_count = FOX_lose_count = ESPN_win_count = ESPN_lose_count = H_In_Or_Out = H_Home_Or_Away = H_Media = H_NBC = H_ABC = H_FOX = H_CBS = H_ESPN = IG_Media = IG_In_Or_Out = IG_Home_Or_Away = 0


def Tree_Calculations(df):

    home_win_count = home_lose_count = away_win_count = away_lose_count = away_count = home_count = in_win_count =        in_lose_count = out_win_count = out_lose_count = in_count = out_count = win_count = lose_count = NBC_count =          ABC_count = ESPN_count = FOX_count = CBS_count = ABC_win_count = ABC_lose_count = NBC_win_count = NBC_lose_count =    CBS_win_count = CBS_lose_count = FOX_win_count = FOX_lose_count = ESPN_win_count = ESPN_lose_count = H_In_Or_Out = H_Home_Or_Away = H_Media = H_NBC = H_ABC = H_FOX = H_CBS = H_ESPN = IG_Media = IG_In_Or_Out = IG_Home_Or_Away = 0

    for row in df.itertuples():
        if 'Is_Home_or_Away' in df.columns:
            col_index = (df.columns.get_loc('Is_Home_or_Away') + 1)
            label_index = (df.columns.get_loc('Label') + 1)

            if row[col_index] == 'Home' and row[label_index] == 'Win':
                home_win_count += 1
                home_count += 1
                win_count += 1
            elif row[col_index] == 'Home' and row[label_index] == 'Lose':
                home_lose_count += 1
                home_count += 1
                lose_count += 1
            elif row[col_index] == 'Away' and row[label_index] == 'Win':
                away_win_count += 1
                away_count += 1
                win_count += 1
            else:
                away_lose_count += 1
                away_count += 1
                lose_count += 1

        if 'Is_Opponent_in_AP25_Preseason' in df.columns:

            col_index = (df.columns.get_loc('Is_Opponent_in_AP25_Preseason') + 1)
            label_index = (df.columns.get_loc('Label') + 1)

            if row[col_index] == 'In' and row[label_index] == 'Win':
                in_win_count += 1
                in_count += 1
                win_count += 1
            elif row[col_index] == 'In' and row[label_index] == 'Lose':
                in_lose_count += 1
                in_count += 1
                lose_count += 1
            elif row[col_index] == 'Out' and row[label_index] == 'Win':
                out_win_count += 1
                out_count += 1
                win_count += 1
            else:
                out_lose_count += 1
                out_count += 1
                lose_count += 1

        if 'Media' in df.columns:

            col_index = (df.columns.get_loc('Media') + 1)
            label_index = (df.columns.get_loc('Label') + 1)

            if row[col_index] == 'NBC':
                if row[label_index] == 'Win':
                    NBC_win_count += 1
                    NBC_count += 1
                    win_count += 1
                else:
                    NBC_lose_count += 1
                    NBC_count += 1
                    lose_count += 1
            elif row[col_index] == 'ABC':
                if row[label_index] == 'Win':
                    ABC_win_count += 1
                    ABC_count += 1
                    win_count += 1
                else:
                    ABC_lose_count += 1
                    ABC_count += 1
                    lose_count += 1
            elif row[col_index] == 'ESPN':
                if row[label_index] == 'Win':
                    ESPN_win_count += 1
                    ESPN_count += 1
                    win_count += 1
                else:
                    ESPN_lose_count += 1
                    ESPN_count += 1
                    lose_count += 1
            elif row[col_index] == 'FOX':
                if row[label_index] == 'Win':
                    FOX_win_count += 1
                    FOX_count += 1
                    win_count += 1
                else:
                    FOX_lose_count += 1
                    FOX_count += 1
                    lose_count += 1
            else:
                if row[label_index] == 'Win':
                    CBS_win_count += 1
                    CBS_count += 1
                    win_count += 1
                else:
                    CBS_lose_count += 1
                    CBS_count += 1
                    lose_count += 1

    if 'Is_Home_or_Away' in df.columns:

        H_Home_Or_Away = H_single(home_count, away_count, home_win_count, home_lose_count) + H_single(away_count, home_count, away_win_count, away_lose_count)
        IG_Home_Or_Away = IG(H_Home_Or_Away, win_count, lose_count)

    if 'Is_Opponent_in_AP25_Preseason' in df.columns:

        H_In_Or_Out = H_single(in_count, out_count, in_win_count, in_lose_count) + H_single(out_count, in_count, out_win_count, out_lose_count)
        IG_In_Or_Out = IG(H_In_Or_Out, win_count, lose_count)

    if 'Media' in df.columns:
        H_NBC = H_single(NBC_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count), NBC_win_count, NBC_lose_count)
        H_ABC = H_single(ABC_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    ABC_win_count, ABC_lose_count)
        H_FOX = H_single(FOX_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    FOX_win_count, FOX_lose_count)
        H_CBS = H_single(CBS_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    CBS_win_count, CBS_lose_count)
        H_ESPN = H_single(ESPN_count, (NBC_count + ABC_count + FOX_count + CBS_count + ESPN_count),    ESPN_win_count, ESPN_lose_count)

        H_Media = H_NBC + H_ABC + H_FOX + H_CBS + H_ESPN
        IG_Media = IG(H_Media, win_count, lose_count)

    d = dict();
    d['H_Home_Or_Away'] = float(H_Home_Or_Away)
    d['IG_Home_Or_Away'] = float(IG_Home_Or_Away)
    d['H_In_Or_Out'] = float(H_In_Or_Out)
    d['IG_In_Or_Out'] = float(IG_In_Or_Out)
    d['H_Media'] = float(H_Media)
    d['IG_Media'] = float(IG_Media)

    return d

def IG_Compare(IG_Media, IG_In_Or_Out, IG_Home_Or_Away, dataframe):

    if IG_Media == IG_In_Or_Out == IG_Home_Or_Away == 0:
        print("Reached End Leaf")
        return 0

    elif max(IG_Media, IG_In_Or_Out, IG_Home_Or_Away) == IG_Media:
        df1 = dataframe.copy()
        df2 = dataframe.copy()
        df3 = dataframe.copy()
        df4 = dataframe.copy()
        df5 = dataframe.copy()
        df1 = Tree_Split(df1, "Media", "NBC")
        df2 = Tree_Split(df2, "Media", "ABC")
        df3 = Tree_Split(df3, "Media", "CBS")
        df4 = Tree_Split(df4, "Media", "FOX")
        df5 = Tree_Split(df5, "Media", "ESPN")

        return [df1, df2, df3, df4, df5]

    elif max(IG_Media, IG_In_Or_Out, IG_Home_Or_Away) == IG_In_Or_Out:
        df6 = dataframe.copy()
        df7 = dataframe.copy()
        df6 = Tree_Split(df6, "Is_Opponent_in_AP25_Preseason", "In")
        df7 = Tree_Split(df7, "Is_Opponent_in_AP25_Preseason", "Out")

        return [df6, df7]

    else:
        df8 = dataframe.copy()
        df9 = dataframe.copy()
        df8 = Tree_Split(df8, "Is_Home_or_Away", "Home")
        df9 = Tree_Split(df9, "Is_Home_or_Away", "Away")

        return [df8, df9]















root = Node("Root")
for row in df.itertuples():
    label_index = (df.columns.get_loc('Label') + 1)
    if row[label_index] == 'Win':
        win_count+=1
    if row[label_index] == 'Lose':
        lose_count+=1

if win_count == 0:
    return lose_node = Node("Lose", parent=root)
if lose_count == 0:
    return win_node = Node("Win", parent=root)

if 'Is_Home_or_Away' not in df.columns and 'Is_Opponent_in_AP25_Preseason' not in df.columns and 'Media' not in df.columns:
    if win_count > lose_count:
        return win_node = Node("Win", parent=root)
    elif win_count < lose_count:
        return lose_node = Node("Lose", parent=root)
    elif win_count == lose_count:
        return win_node = Node("Win", parent=root)

tree_dict = Tree_Calculations(df)
branch1 = Node("Branch1")

branch = IG_Compare(tree_dict["IG_Media"], tree_dict["IG_In_Or_Out"], tree_dict["IG_Home_Or_Away"], df)

#branches split by media

for i in branch1:
    tree_dict = Tree_Calculations(i)
    branch2 = IG_Compare(tree_dict["IG_Media"], tree_dict["IG_In_Or_Out"], tree_dict["IG_Home_Or_Away"], i)

