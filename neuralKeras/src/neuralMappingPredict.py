# import setuptools
import os
# import sys
import numpy as np

# import keras
from keras.models import Sequential, load_model

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./pred/matrix")

print('> Readind matrix data...')
for file in matrixFiles:
    fileIn = open("./matrix/" + file, "r")

    matrix = fileIn.readlines()

    for i in range(len(matrix)):
        matrix[i] = matrix[i][:-2].split(' ')
        for j in range(len(matrix[i])):
            matrix[i][j] = int(matrix[i][j])

        pairList = []
        for j in range(len(matrix[i])):
            pairList.append((j, matrix[i][j]))

        pairList.sort(key=lambda x: x[1])

        for j in range(len(matrix[i])):
            if (pairList[j][1] == 0):
                matrix[i][pairList[j][0]] = 0
            else:
                matrix[i][pairList[j][0]] = j + 1

    tmp = np.array(matrix)
    tmp = np.expand_dims(tmp, axis=2)
    matrixList.append(tmp)

    fileIn.close()

matrixVec = np.array(matrixList)
matrixDim = int(matrixVec.shape[1])
numOfSet = int(matrixVec.shape[0])

# print(matrixVec.shape)
# for i in range(matrixDim):
#     print(i)
#     for j in range(matrixDim):
#         print(matrixVec[0][i][j], end=' ')
#     print('\n')

# sys.exit(0)

print('> Preparing for prediction...')
lenMapStr = 4
model = Sequential()
model = load_model('./nets/net1.h5')

max = 0
if (matrixDim <= 8):
    max = 2
elif ((matrixDim <= 16) or (matrixDim <= 32) or (matrixDim <= 64)):
    max = 4
elif ((matrixDim <= 128) or (matrixDim <= 256) or (matrixDim <= 512)):
    max = 8
elif ((matrixDim <= 1024) or (matrixDim <= 2048)):
    max = 16


print('> Predict on test data...')
for i in range(len(matrixVec)):
    pred = model.predict(matrixVec[i:i + 1])
    print("./pred/prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

    for j in range(matrixDim):
        for k in range(lenMapStr):
            if (abs(pred[0][j * lenMapStr + k] - 0) > 0.00001):
                tmp = 1 / pred[0][j * lenMapStr + k]
                if (round(tmp) > max - 1):
                    fileOut.write(str(int(0)) + ' ')
                else:
                    fileOut.write(str(int(round(tmp))) + ' ')
            else:
                fileOut.write(str(int(0)) + ' ')
        fileOut.write('\n')

    fileOut.close()

# score = model.evaluate(matrixVec, mappingVec, batch_size=3)
# print(score)
