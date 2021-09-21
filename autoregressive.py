from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convertDataset(data,window):
    inputList = []
    outputList = []
    for i in range(window):
        inputList.append(data[i:(len(data) - window * 2 + 1 + i)])
        outputList.append(data[i + window:(len(data) - window * 2 + 1 + i + window)])

    return (np.array(inputList).reshape(-1, window), np.array(outputList).reshape(-1, window))


myWindow = 3

epochs = 999999
validation_split = 0.2
verbose = 1 # 0 silent, 1 progress, 2 text
shuffle = True
patience = 10
es = EarlyStopping(monitor='loss', mode='auto', verbose=verbose, patience=patience, restore_best_weights=True)

# trainingData = pd.read_csv("E:/Baris_Dosyalar/Hacettepe/Tez/ER12.csv", sep = ";")
# t1 = trainingData["T1"].to_numpy()

t1 = np.linspace(0, 49, 50)

(inputData,outputData) = convertDataset(t1,myWindow)

i = Input(shape = myWindow)
x = Dense(myWindow)(i)
model = Model(i,x)
model.compile(loss="mse",optimizer="adam")
history = model.fit(inputData,outputData,epochs=epochs,validation_split=validation_split, verbose = verbose, shuffle = shuffle, callbacks=[es])

plt.plot(history.history["loss"][1:-1], label="loss")
plt.plot(history.history["val_loss"][1:-1], label="val_loss")
plt.legend()
plt.show()

# testData = pd.read_csv("E:/Baris_Dosyalar/Hacettepe/Tez/ER13.csv", sep = ";")
# t1_test = testData["T1"].to_numpy()

t1_test = np.linspace(50,99,50)

(inputTestData,outputTestData) = convertDataset(t1_test,myWindow)

predictedOutput = model.predict(inputTestData)

for i in range(myWindow):
    plt.plot(outputTestData[i,:],label="target")
    plt.plot(predictedOutput[i,:],label="predicted")
    plt.legend()
    plt.show()


print("End of Code")