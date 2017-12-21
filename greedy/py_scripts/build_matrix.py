import sys
import numpy as np


file_out = open(sys.argv[1], 'w')
matrix = np.zeros((int(sys.argv[2]), int(sys.argv[2])))

for j in range(3, int(sys.argv[2]) + 3):
    file_in = open(sys.argv[j], 'r')

    proc = int(file_in.readline())

    # tmp = file_in.read().split('\n')

    for line in file_in:
        tmp = line.split()
        matrix[proc, int(tmp[0])] += int(tmp[1])

    file_in.close()


for i in range(0, int(sys.argv[2])):
    for j in range(0, int(sys.argv[2])):
        file_out.write(str(int(matrix[i][j])))
        file_out.write(' ')
    file_out.write('\n')

file_out.close()
