import sys


file_out = open("matrix_cor", 'w')

file_in = open(sys.argv[1], 'r')

end = int(sys.argv[2])

for j in range(0, end):
    tmp = file_in.readline().split()
    for i in range(0, end):
        res = tmp[i].split(".")
        file_out.write(res[0] + ' ')
    file_out.write('\n')


file_in.close()
file_out.close()
