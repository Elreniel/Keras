from tensorflow.keras.layers import Input, Dense, SimpleRNN, GRU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convertDataset(data, window):
    inputList = []
    outputList = []
    for i in range(window):
        inputList.append(data[i:(len(data) - window * 2 + 1 + i)])
        outputList.append(data[i + window:(len(data) - window * 2 + 1 + i + window)])

    return (np.array(inputList).reshape(-1, window, 1), np.array(outputList).reshape(-1, window))


myWindow = 16

epochs = 999999
validation_split = 0.2
verbose = 1 # 0 silent, 1 progress, 2 text
shuffle = False
patience = 5
es = EarlyStopping(monitor='loss', mode='auto', verbose=verbose, patience=patience, restore_best_weights=True)

trainingData = pd.read_csv("E:/Baris_Dosyalar/Hacettepe/Tez/ER12.csv", sep=";")
t1 = trainingData["T1"].to_numpy()

# t1 = np.linspace(0, 49, 50)

(inputData, outputData) = convertDataset(t1, myWindow)

testData = pd.read_csv("E:/Baris_Dosyalar/Hacettepe/Tez/ER13.csv", sep=";")
t1_test = testData["T1"].to_numpy()

# t1_test = np.linspace(50,99,50)

(inputTestData, outputTestData) = convertDataset(t1_test, myWindow)

testResults = []
for modelType in range(3):
    i = Input(shape=(myWindow, 1))
    if modelType == 0:
        x = SimpleRNN(10)(i)
    elif modelType == 1:
        x = GRU(10)(i)
    elif modelType == 2:
        x = LSTM(10)(i)

    x = Dense(myWindow)(x)

    model = Model(i, x)
    model.compile(loss="mse", optimizer="adam")
    history = model.fit(inputData, outputData, epochs=epochs, validation_split=validation_split, verbose=verbose, shuffle=shuffle, callbacks=[es])

    # plt.plot(history.history["loss"][1:-1], label="loss")
    # plt.plot(history.history["val_loss"][1:-1], label="val_loss")
    # plt.legend()
    # plt.show()

    testResults.append(model.evaluate(inputTestData, outputTestData, verbose=verbose))

for modelType in range(3):
    if modelType == 0:
        tempStr = "RNN Result: "
    elif modelType == 1:
        tempStr = "GRU Result: "
    elif modelType == 2:
        tempStr = "LSTM Result: "

    print(tempStr + str(testResults[modelType]))

print("End of Code")