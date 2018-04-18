import sys
import numpy as np

from keras.models import Sequential, load_model
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

# <matrixFile>
if (len(sys.argv) != 2):
    print('> Unexpected quantity of arguments, check your comand string.')
    sys.exit(1)

classList = ['TRIPLE INVERSE DIAGONAL', 'DOUBLE DIRECT DIAGONAL',
             'DOUBLE DIRECT + DOUBLE INVERSE DIAGONALS', 'UPPER TIANGLE',
             'DOUBLE INVERSE DIAGONALS', 'LOWWER TIANGLE', 'RANDOM']

matrixList = []

fileIn = open(sys.argv[1], "r")

matrix = fileIn.readlines()
dim = len(matrix)

print('> Readind matrix data...')
for i in range(dim):
    matrix[i] = matrix[i][:-1].split(' ')
    # for j in range(len(matrix[i])):
    #     matrix[i][j] = int(matrix[i][j])

    pairList = []
    for j in range(dim):
        pairList.append((j, int(matrix[i][j])))

    pairList.sort(key=lambda x: x[1])

    for j in range(dim):
        if (pairList[j][1] == 0):
            matrix[i][pairList[j][0]] = 0
        else:
            matrix[i][pairList[j][0]] = j + 1

# plt.imshow(matrix)
# plt.colorbar()
# plt.show()

tmp = np.array(matrix)
tmp = np.expand_dims(tmp, axis=2)
matrixList.append(tmp)

fileIn.close()

matrixVec = np.array(matrixList)
matrixDim = int(matrixVec.shape[1])
numOfSet = int(matrixVec.shape[0])

print(matrixVec.shape)

print('> Preparing for prediction...')
model = Sequential()
# model = load_model('./nets/net1.h5')
model = load_model('./nets/goodNet3.h5')

print('> Predict on test data...')
pred = model.predict(matrixVec[0:1])
res = str(np.where(pred == pred.max())[1][0])

print('> Probabilities of prediction are:')
print(pred)
resMsg = '> Your matrix class is {} ({})'.format(res, classList[int(res)])
print(resMsg)
