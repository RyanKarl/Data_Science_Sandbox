#Ryan Karl

import pandas
import math
import numpy

#Function to recursively print patterns
def print_patterns(C1, check):
    check = False

    #Find support values
    F1 = check_support(C1)

    #Print all sequences with enough support
    for key, values in F1.items():
        if values >= 2:
            print(key, values)
            check = True
        else:
            del F1[key]

    #Continue recursively printing until min_sup not adequate
    if check == True:
        C2 = itemset_gen(F1)
        return print_patterns(C2, check)

    else:
        return check

#Partly inspired by https://www.geeksforgeeks.org/given-two-strings-find-first-string-      subsequence-second/
#Function to recursively find if a string is a substring of another string
def is_sub_sequence(stringfirst, stringsecond, m, n):
    if m == 0:
        return True
    if n == 0:
        return False

    #Compare each character from each string
    if stringfirst[m-1] == stringsecond[n-1]:
        return is_sub_sequence(stringfirst, stringsecond, m-1, n-1)

    return is_sub_sequence(stringfirst, stringsecond, m, n-1)

#Function to generate new candidate patterns
def itemset_gen(frequent_itemsets):
    temp_itemset1 = []
    temp_itemset2 = []
    new_itemset = {}

    #Get previous patterns
    for i in frequent_itemsets.keys():
        temp_itemset1.append(i)

    #Append new characters to new potential patterns
    for j in temp_itemset1:
        last_char = j[-1:]
        while last_char != 'e':
            temp_itemset2.append(j+chr(ord(last_char) + 1))
            last_char = chr(ord(last_char) + 1)

    #Reset dictionary count
    for x in temp_itemset2:
        new_itemset[x] = 0

    return new_itemset

#Function to count the support of each pattern
def check_support(support_dict):
    sequence_char_list = []

    #Loop over transactions and candidate patterns to check for overlap
    for current_key in support_dict.keys():

        for row in df.itertuples(index=False):

            row_char_sequence = ""

            for val in row:
                if str(val) != 'nan':
                    row_char_sequence = row_char_sequence + str(val)
            #Check if subsequence exists and increment counter if yes
            if is_sub_sequence(str(current_key), row_char_sequence, len(str(current_key)), len(row_char_sequence)):
                support_dict[current_key] += 1

    #Remove sequences without enough support
    for key, count in support_dict.items():
        if count < 2:
            support_dict.pop('key', None)

    return support_dict

#Driver code to read file and preprocess
df = pandas.read_csv('Q1.txt', sep='\t')
df = df.drop('ID', axis = 1)
check = False
C1 = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
check = print_patterns(C1, check)

