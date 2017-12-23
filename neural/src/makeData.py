import subprocess
import sys
import time


def command(com):
    result = subprocess.Popen(com.split(), stdout=subprocess.PIPE)
    result.wait()
    result = result.communicate()[0]
    return result[:-1]


if (len(sys.argv) != 5):
    err = "Unexpected quantity of arguments, check your comand string:\n"
    err += "<matrixDir> <mappingDir> <numOfFiles> <matrixDim>"
    print(err)
    sys.exit(1)

num = int(sys.argv[3])
matrixDim = int(sys.argv[4])

for i in range(num):
    com = './bin/com_matrix_gen {}{}{} {}'.format(
        sys.argv[1], "/matrix", i + 1, matrixDim)

    command(com)

    com = './bin/greedy {}{}{} {} '.format(
        sys.argv[1], "/matrix", i + 1, matrixDim)
    com += '{}{}{}'.format(sys.argv[2], "/mapping", i + 1)

    command(com)

    time.sleep(1)
