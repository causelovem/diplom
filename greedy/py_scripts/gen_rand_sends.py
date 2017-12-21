import numpy as np
import random as rand
import sys


rand.seed()

file = open(sys.argv[1], 'w')

flag = 1

num_str = int(sys.argv[2])
mas = np.empty((num_str, 2))

for i in range(0, num_str):
    mas[i][0] = -1
    mas[i][1] = -1

for i in range(0, num_str):
    flag = 1
    tmp = int(rand.randint(0, num_str - 1))
    while (flag == 1):
        flag = 0
        for j in range(0, i):
            if (mas[j][0] == tmp) | (tmp == i):
                flag = 1
                break

        if (flag == 1):
            tmp = int(rand.randint(0, num_str - 1))

    mas[i][0] = tmp
    mas[tmp][1] = i

for i in range(0, num_str):
    file.write(str(int(mas[i][0])))
    file.write(' ')
    file.write(str(int(mas[i][1])))
    file.write('\n')

file.close()
