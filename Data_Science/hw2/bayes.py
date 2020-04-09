import pandas
import math
df = pandas.read_csv('Dataset-football-train.txt', sep='\t')#, lineterminator='\r')
df = df.drop(columns=['ID', 'Date', 'Opponent'])
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from itertools import cycle

def bayes_calculations(data):
    home_win_count = home_lose_count = away_win_count = away_lose_count = away_count = home_count = in_win_count =        in_lose_count = out_win_count = out_lose_count = in_count = out_count = win_count = lose_count = NBC_count =          ABC_count = ESPN_count = FOX_count = CBS_count = ABC_win_count = ABC_lose_count = NBC_win_count = NBC_lose_count =    CBS_win_count = CBS_lose_count = FOX_win_count = FOX_lose_count = ESPN_win_count = ESPN_lose_count = H_In_Or_Out =    H_Home_Or_Away = H_Media = H_NBC = H_ABC = H_FOX = H_CBS = H_ESPN = IG_Media = IG_In_Or_Out = IG_Home_Or_Away = GR_Media = GR_In_Or_Out = GR_Home_Or_Away = 0

    _, n_columns = data.shape

    for column_index in range(n_columns - 1):        # excluding the last column which is the label
        values = data[:, column_index]
        label_index = -1

        for item in values:
            label_index += 1

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

    d = dict()

    d['pp_win'] = win_count/float(win_count + lose_count)
    d['pp_lose'] = lose_count/float(win_count + lose_count)

    d['lp_NBC_win'] = NBC_win_count/float(win_count)
    d['lp_NBC_lose'] = NBC_lose_count/float(lose_count)

    d['lp_ABC_win'] = ABC_win_count/float(win_count)
    d['lp_ABC_lose'] = ABC_lose_count/float(lose_count)

    d['lp_CBS_win'] = CBS_win_count/float(win_count)
    d['lp_CBS_lose'] = CBS_lose_count/float(lose_count)

    d['lp_FOX_win'] = FOX_win_count/float(win_count)
    d['lp_FOX_lose'] = FOX_lose_count/float(lose_count)

    d['lp_ESPN_win'] = ESPN_win_count/float(win_count)
    d['lp_ESPN_lose'] = ESPN_lose_count/float(lose_count)

    d['lp_Home_win'] = home_win_count/float(win_count)
    d['lp_Home_lose'] = home_lose_count/float(lose_count)

    d['lp_Away_win'] = away_win_count/float(win_count)
    d['lp_Away_lose'] = away_lose_count/float(lose_count)

    d['lp_In_win'] = in_win_count/float(win_count)
    d['lp_In_lose'] = in_lose_count/float(lose_count)

    d['lp_Out_win'] = out_win_count/float(win_count)
    d['lp_Out_lose'] = out_lose_count/float(lose_count)

    return d

def bayes_algorithm(df):

    # data preparations
    data = df.values
    data_dict = bayes_calculations(data)

    return data_dict


classifier = bayes_algorithm(df)

df_test = pandas.read_csv('Dataset-football-test.txt', sep='\t')#, lineterminator='\r')
df_test = df_test.drop(columns=['ID', 'Date', 'Opponent'])

true_positive = true_negative = false_positive = false_negative = 0

for row in df_test.itertuples():

    row_media_win = str('lp_' + row[3] + '_win')
    row_in_or_out_win = str('lp_' + row[2] + '_win')
    row_home_or_away_win = str('lp_' + row[1] + '_win')

    row_media_lose = str('lp_' + row[3] + '_lose')
    row_in_or_out_lose = str('lp_' + row[2] + '_lose')
    row_home_or_away_lose = str('lp_' + row[1] + '_lose')

    win_p = classifier['pp_win'] * classifier[row_media_win] * classifier[row_in_or_out_win] * classifier[row_home_or_away_win]
    lose_p = classifier['pp_lose'] * classifier[row_media_lose] * classifier[row_in_or_out_lose] * classifier[row_home_or_away_lose]

    if win_p >= lose_p:
        prediction = 'Win'
    else:
        prediction = 'Lose'

    print(prediction)

    if row[4] == prediction and prediction == "Win":
        true_positive += 1
    elif row[4] == prediction and prediction == "Lose":
        true_negative += 1
    elif row[4] != prediction and prediction == "Win":
        false_positive += 1
    else:
        false_negative += 1

print("Accuracy = " + str((true_positive+true_negative)/float((true_negative+true_positive+false_negative+             false_positive))))
print("Precision = " +str((true_positive)/float((true_positive+false_positive))))
print("Recall = " +str((true_positive)/float((true_positive+false_negative))))
precision = (true_positive)/float((true_positive+false_positive))
recall = (true_positive)/float((true_positive+false_negative))
print("F1 = " +str((precision*recall)/(precision+recall)))

