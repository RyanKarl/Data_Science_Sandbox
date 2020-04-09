import pandas
import math
df = pandas.read_csv('Dataset-football-train.txt', sep='\t')#, lineterminator='\r')
df = df.drop(columns=['ID', 'Date', 'Opponent'])
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from itertools import cycle

tree_traversal = []

def classify_data(data):

    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    if len(unique_classes) == 1:
        #print(unique_classes[0])
        tree_traversal.append(unique_classes[0])
        return unique_classes[0]

    if counts_unique_classes[0] == counts_unique_classes[1]:
        classification = unique_classes[1]
        #print(classification)
        tree_traversal.append(classification)
        return classification

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    #print(classification)
    tree_traversal.append(classification)
    return classification

def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

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

def tree_calculations(data):
    home_win_count = home_lose_count = away_win_count = away_lose_count = away_count = home_count = in_win_count =        in_lose_count = out_win_count = out_lose_count = in_count = out_count = win_count = lose_count = NBC_count =          ABC_count = ESPN_count = FOX_count = CBS_count = ABC_win_count = ABC_lose_count = NBC_win_count = NBC_lose_count =    CBS_win_count = CBS_lose_count = FOX_win_count = FOX_lose_count = ESPN_win_count = ESPN_lose_count = H_In_Or_Out =    H_Home_Or_Away = H_Media = H_NBC = H_ABC = H_FOX = H_CBS = H_ESPN = IG_Media = IG_In_Or_Out = IG_Home_Or_Away = 0

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

    return d

def Tree_Split(data, search_string):
    val_index = -1
    new_data = []
    for val in data:
        val_index += 1
        if search_string in val:
            new_data.append(val)

    final_data = []
    for item in new_data:
        index = np.argwhere(item==search_string)
        item = np.delete(item, index)
        final_data.append(item)

    final_data = np.asarray(final_data)
    return(final_data)

def IG_Compare(d, data):

    if max(d["IG_Media"], d["IG_In_Or_Out"], d["IG_Home_Or_Away"]) == d["IG_Media"]:
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

        return [data1, data2, data3, data4, data5], "Media", ["NBC", "ABC", "CBS", "FOX", "ESPN"], True

    elif max(d["IG_Media"], d["IG_In_Or_Out"], d["IG_Home_Or_Away"]) == d["IG_In_Or_Out"]:
         data6 = data.copy()
         data7 = data.copy()
         data6 = Tree_Split(data6, "In")
         data7 = Tree_Split(data7, "Out")
         return [data6, data7], "Is_Opponent_in_AP25_Preseason", ["In", "Out"], True

    else:
         data8 = data.copy()
         data9 = data.copy()
         data8 = Tree_Split(data8, "Home")
         data9 = Tree_Split(data9, "Away")

         return [data8, data9], "Is_Home_or_Away", ["Home", "Away"], True

def decision_tree_algorithm(df, counter=0, max_depth=3):

    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
        _, n_columns = data.shape
    else:
        data = df

    #print("Level " + str(counter))
    tree_traversal.append("Level " + str(counter))
    data_dict = tree_calculations(data)

    # base cases
    if (check_purity(data)) or (counter == max_depth) or (data_dict["IG_Media"] + data_dict["IG_In_Or_Out"] + data_dict["IG_Home_Or_Away"]) == 0:
        classification = classify_data(data)

        return classification

    # recursive part
    else:
        counter += 1

        # helper functions
        #split_column, split_value = determine_best_split(data, potential_splits)
        data_dict = tree_calculations(data)

        #split_list has data above and below
        split_list, split_column, split_value, stop_condition = IG_Compare(data_dict, data)

        #data_below, data_above = split_data(data, split_column, split_value)

        #print(split_column)
        #print(split_value)
        tree_traversal.append(split_column)
        tree_traversal.append(split_value)

        for item in split_value:
            question = "{} <= {}".format(split_column, item)
            sub_tree = {question: []}

        # instantiate sub-tree
        #feature_name = split_column
        #question = "{} <= {}".format(feature_name, split_value)
        #sub_tree = {question: []}

        # find answers (recursion)
        if split_column != "Media":

            yes_answer = decision_tree_algorithm(split_list[0], counter, max_depth)
            no_answer = decision_tree_algorithm(split_list[1], counter, max_depth)
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

        elif split_column == "Media":

            NBC_answer = decision_tree_algorithm(split_list[0], counter, max_depth)
            ABC_answer = decision_tree_algorithm(split_list[1], counter, max_depth)
            CBS_answer = decision_tree_algorithm(split_list[2], counter, max_depth)
            FOX_answer = decision_tree_algorithm(split_list[3], counter, max_depth)
            ESPN_answer = decision_tree_algorithm(split_list[4], counter, max_depth)

            if NBC_answer == ABC_answer == CBS_answer == FOX_answer == ESPN_answer:
                sub_tree = NBC_answer
            else:
                sub_tree[question].append(NBC_answer)
                sub_tree[question].append(ABC_answer)
                sub_tree[question].append(CBS_answer)
                sub_tree[question].append(FOX_answer)
                sub_tree[question].append(ESPN_answer)

        return sub_tree

tree = decision_tree_algorithm(df, max_depth=3)

a0 = []
q0 = []
c0 = []
a1 = []
q1 = []
c1 = []
a2 = []
q2 = []
c2 = []
a3 = []
q3 = []
c3 = []

for a, name in enumerate(tree_traversal, start=0):
    if name == "Level 0":
        if tree_traversal[a+1] == "Win" or tree_traversal[a+1] == "Lose":
            a0.append(tree_traversal[a+1])
        else:
            q0.append(tree_traversal[a+1])
            a0.append(tree_traversal[a+2])
            #c0.append(False)

    elif name == "Level 1":
        if tree_traversal[a+1] == "Win" or tree_traversal[a+1] == "Lose":
            a1.append(tree_traversal[a+1])
        else:
            q1.append(tree_traversal[a+1])
            a1.append(tree_traversal[a+2])
            #c1.append(False)

    elif name == "Level 2":
        if tree_traversal[a+1] == "Win" or tree_traversal[a+1] == "Lose":
            a2.append(tree_traversal[a+1])
        else:
            q2.append(tree_traversal[a+1])
            a2.append(tree_traversal[a+2])
            #c2.append(False)

    elif name == "Level 3":
        if tree_traversal[a+1] == "Win" or tree_traversal[a+1] == "Lose":
            a3.append(tree_traversal[a+1])
        else:
            q3.append(tree_traversal[a+1])
            a3.append(tree_traversal[a+2])
            #c3.append(False)

#print(tree_traversal)

print("q0 = " + str(q0))
print("a0 = " + str(a0))
#print(c0)
print("q1 = " + str(q1))
print("a1 = " + str(a1))
#print(c1)
print("q2 = " + str(q2))
print("a2 = " + str(a2))
#print(c2)
print("q3 = " + str(q3))
print("a3 = " + str(a3))
#print(c3)


df_test = pandas.read_csv('Dataset-football-test.txt', sep='\t')#, lineterminator='\r')
df_test = df_test.drop(columns=['ID', 'Date', 'Opponent'])
df_test_index = {
  "Is_Home_or_Away": 1,
  "Is_Opponent_in_AP25_Preseason": 2,
  "Media": 3
}

true_positive = true_negative = false_positive = false_negative = 0


for row in df_test.itertuples():

    row_media = row[3]
    row_in_or_out = row[2]
    row_home_or_away = row[1]

    if q0[0] == "Media":

        a0_position = a0[0].index(row_media)

        if a1[a0_position] == 'Win' or a1[a0_position] == 'Lose':

            prediction = a1[a0_position]

        elif a1[a0_position] == ['In', 'Out']:

            p = a0_position - 1
            branch_count = 0
            while p >= 0:
                if a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count +=1
                p-=1

            a1_position = a1[a0_position].index(row_in_or_out)


            if a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Win' or a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position+(len(a1[a0_position]) * branch_count)]

            else:

                p = a0_position-1
                branch_count = 0
                while p >= 0:
                    if a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                              #a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position+(len(a1[a0_position])* branch_count)].index(row_home_or_away)

                prediction = a3[a2_position+(len(a2[a1_position]) * branch_count)]

        else:

            p = a0_position-1
            branch_count = 0
            while p >= 0:
                if a0_position == 0:
                    branch_count = 0
                elif a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count += 1
                p -= 1

            a1_position = a1[a0_position].index(row_home_or_away)

            if a2[a1_position + (len(a1[a0_position]) * branch_count)] == 'Win' or a2[
                a1_position + (len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position + (len(a1[a0_position]) * branch_count)]

            else:

                p = a1_position-1
                branch_count = 0
                while p >= 0:
                    if a1_position == 0:
                        branch_count = 0
                    elif a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                    # a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position + (len(a1[a0_position]) * branch_count)].index(row_in_or_out)

                prediction = a3[a2_position + (len(a2[a1_position]) * branch_count)]











    if q0[0] == "Is_Opponent_in_AP25_Preseason":

        a0_position = a0[0].index(row_in_or_out)

        if a1[a0_position] == 'Win' or a1[a0_position] == 'Lose':

            prediction = a1[a0_position]

        elif a1[a0_position] == ['Home', 'Away']:

            p = a0_position - 1
            branch_count = 0
            while p >= 0:
                if a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count +=1
                p-=1

            a1_position = a1[a0_position].index(row_home_or_away)


            if a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Win' or a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position+(len(a1[a0_position]) * branch_count)]

            else:

                p = a0_position-1
                branch_count = 0
                while p >= 0:
                    if a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                              #a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position+(len(a1[a0_position])* branch_count)].index(row_media)

                prediction = a3[a2_position+(len(a2[a1_position]) * branch_count)]

        else:

            p = a0_position-1
            branch_count = 0
            while p >= 0:
                if a0_position == 0:
                    branch_count = 0
                elif a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count += 1
                p -= 1

            a1_position = a1[a0_position].index(row_media)

            if a2[a1_position + (len(a1[a0_position]) * branch_count)] == 'Win' or a2[
                a1_position + (len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position + (len(a1[a0_position]) * branch_count)]

            else:

                p = a1_position-1
                branch_count = 0
                while p >= 0:
                    if a1_position == 0:
                        branch_count = 0
                    elif a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                    # a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position + (len(a1[a0_position]) * branch_count)].index(row_home_or_away)

                prediction = a3[a2_position + (len(a2[a1_position]) * branch_count)]


    if q0[0] == "Is_Home_Or_Away":

        a0_position = a0[0].index(row_home_or_away)

        if a1[a0_position] == 'Win' or a1[a0_position] == 'Lose':

            prediction = a1[a0_position]

        elif a1[a0_position] == ['In', 'Out']:

            p = a0_position - 1
            branch_count = 0
            while p >= 0:
                if a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count +=1
                p-=1

            a1_position = a1[a0_position].index(row_in_or_out)


            if a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Win' or a2[a1_position+(len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position+(len(a1[a0_position]) * branch_count)]

            else:

                p = a0_position-1
                branch_count = 0
                while p >= 0:
                    if a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                              #a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position+(len(a1[a0_position])* branch_count)].index(row_media)

                prediction = a3[a2_position+(len(a2[a1_position]) * branch_count)]

        else:

            p = a0_position-1
            branch_count = 0
            while p >= 0:
                if a0_position == 0:
                    branch_count = 0
                elif a1[p] != 'Win' and a1[p] != 'Lose':
                    branch_count += 1
                p -= 1

            a1_position = a1[a0_position].index(row_media)

            if a2[a1_position + (len(a1[a0_position]) * branch_count)] == 'Win' or a2[
                a1_position + (len(a1[a0_position]) * branch_count)] == 'Lose':

                prediction = a2[a1_position + (len(a1[a0_position]) * branch_count)]

            else:

                p = a1_position-1
                branch_count = 0
                while p >= 0:
                    if a1_position == 0:
                        branch_count = 0
                    elif a2[p] != 'Win' and a2[p] != 'Lose':
                        branch_count += 1
                    p -= 1

                    # a2[a1_position + (len(a1[a0_position]) * branch_count)]
                a2_position = a2[a1_position + (len(a1[a0_position]) * branch_count)].index(row_in_or_out)

                prediction = a3[a2_position + (len(a2[a1_position]) * branch_count)]


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


