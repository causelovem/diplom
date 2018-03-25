# import setuptools
import os
# import sys
import numpy as np

# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
# from keras.optimizers import SGD

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./matrix")

print('> Readind matrix data...')
for file in matrixFiles:
    fileIn = open("./matrix/" + file, "r")

    matrix = fileIn.readlines()
    dim = len(matrix)

    for i in range(dim):
        matrix[i] = matrix[i][:-2].split(' ')
        for j in range(len(matrix[i])):
            matrix[i][j] = int(matrix[i][j])

        pairList = []
        for j in range(dim):
            pairList.append((j, matrix[i][j]))

        pairList.sort(key=lambda x: x[1])

        for j in range(dim):
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

mappingList = []
mappingFiles = os.listdir("./mapping")

print('> Readind mapping data...')
for file in mappingFiles:
    fileIn = open("./mapping/" + file, "r")

    mapping = fileIn.readlines()
    dim = len(mapping)

    for i in range(dim):
        mapping[i] = mapping[i][:-1].split(' ')
        strDim = len(mapping[i])
        for j in range(strDim):
            if (mapping[i][j] != '0'):
                mapping[i][j] = 1.0 / int(mapping[i][j])
            else:
                mapping[i][j] = int(mapping[i][j])
    tmp = np.array(mapping)
    mappingList.append(tmp)

    fileIn.close()

mappingVec = np.array(mappingList)
mappingVec = mappingVec.reshape(numOfSet, matrixDim * 4)

print('> Preparing for train...')
lenMapStr = 4
numFilt = 64
model = Sequential()
# model.add(Conv2D(numFilt, (4, 4), padding='same',
#                  input_shape=(matrixDim, matrixDim, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(numFilt, (4, 4), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(numFilt, (4, 4), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(int(matrixDim * matrixDim * numFilt / 64),
#                 activation='sigmoid'))
# model.add(Dense(matrixDim * lenMapStr, activation='sigmoid'))


model.add(ZeroPadding2D((1, 1), input_shape=(matrixDim, matrixDim, 1)))
# model.add(Conv2D(numFilt, (3, 3), padding='same',
#                  input_shape=(matrixDim, matrixDim, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
# model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(numFilt, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
# model.add(Dense(int(matrixDim * matrixDim * numFilt / 32),
#                 activation='sigmoid'))
model.add(Dense(int(matrixDim * matrixDim * numFilt / (64 * 64)),
                activation='sigmoid'))
# model.add(Dense(int(matrixDim * matrixDim / 64),
#                 activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(matrixDim * lenMapStr, activation='sigmoid'))

# model.add(Dense(int(matrixDim * matrixDim * numFilt / 64),
#                 activation='sigmoid', init='he_normal'))
# model.add(Dense(matrixDim * lenMapStr, activation='sigmoid',
#                 init='he_normal'))

# softplus softsign relu sigmoid/hard_sigmoid

# Adadelta Adam sgd
# poisson mse logcosh mean_squared_logarithmic_error categorical_hinge
# sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='Adadelta', metrics=['accuracy'])

model.fit(matrixVec, mappingVec, epochs=5000, batch_size=5)
score = model.evaluate(matrixVec, mappingVec, batch_size=5)

model.save('./nets/net2.h5')

max = 0
if (matrixDim <= 8):
    max = 2
elif ((matrixDim <= 16) or (matrixDim <= 32) or (matrixDim <= 64)):
    max = 4
elif ((matrixDim <= 128) or (matrixDim <= 256) or (matrixDim <= 512)):
    max = 8
elif ((matrixDim <= 1024) or (matrixDim <= 2048)):
    max = 16


print('> Predict on trannig data...')
for i in range(len(matrixVec)):
    pred = model.predict(matrixVec[i:i + 1])
    print("./prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./prediction/mapping" + str(i + 1) + "Pred", "w")

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

print(score)
