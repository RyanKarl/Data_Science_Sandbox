import math

list = [0, 5]
list_sum = sum(list)
if list_sum == 0:
     final = 0

final = 0
for i in list:
    i = float(i)
    if i == 0:
        final += 0
    else:
        final += (((-i/list_sum))*(math.log((i/list_sum), 2)))

print(final)
