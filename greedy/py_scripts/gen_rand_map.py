import numpy as np
import random as rand


rand.seed()

flag = 1

num_str = 512
map = np.zeros((num_str, 4))

for i in range(0, num_str):
    flag = 1
    map[i][0] = int(rand.randint(0, 7))
    map[i][1] = int(rand.randint(0, 7))
    map[i][2] = int(rand.randint(0, 7))
    while (flag == 1):
        flag = 0
        for j in range(0, i):
            if (map[i][0] == map[j][0]) & (map[i][1] == map[j][1]) & (map[i][2] == map[j][2]):
                flag = 1
                break

        if (flag == 1):
            map[i][0] = int(rand.randint(0, 7))
            map[i][1] = int(rand.randint(0, 7))
            map[i][2] = int(rand.randint(0, 7))

file = open('yura.map', 'w')

for i in range(num_str):
    file.write(str(int(map[i][0])))
    file.write(' ')
    file.write(str(int(map[i][1])))
    file.write(' ')
    file.write(str(int(map[i][2])))
    file.write(' ')
    file.write(str(int(map[i][3])))
    file.write('\n')

file.close()
