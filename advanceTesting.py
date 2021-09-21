# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:56:25 2021

@author: bcosk
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.objectives import mean_squared_error, mean_absolute_error
from keras.layers import LSTM, Flatten
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from random import uniform
import xlsxwriter

def dataPreparing(dataFirst,colNumber):
    
    newColumnNames = []
    for i in range(0,len(dataFirst.columns)):
        newColumnNames.append(dataFirst.columns[i].replace("T","Q"))
        
    confidenceFirst = pd.DataFrame(np.zeros((len(dataFirst),len(dataFirst.columns))), columns = newColumnNames)
    
    inputData = dataFirst.to_numpy()
    outputData = confidenceFirst.to_numpy()
    outputData1 = dataFirst.to_numpy()
        
    processedDataFirst = dataFirst.copy()
    processedconfidenceFirst = confidenceFirst.copy()
    for i in range(0,len(processedDataFirst)):
        tempConfidence = (i%201 - 100)/100
        processedDataFirst[processedDataFirst.columns[colNumber]][i] += processedDataFirst[processedDataFirst.columns[colNumber]][i] * tempConfidence
        processedconfidenceFirst[processedconfidenceFirst.columns[colNumber]][i] += tempConfidence
            
    return processedDataFirst, processedconfidenceFirst[processedconfidenceFirst.columns[colNumber]], dataFirst[dataFirst.columns[colNumber]]

def randomNoiseGenerator(data,confidence,selectedData):
    processedData = data.copy()
    processedConfidence = confidence.copy()
    for i in range(0,len(processedData[processedData.columns[selectedData]])):
        tempConfidence = uniform(-1,1)
        processedData[processedData.columns[selectedData]][i] += processedData[processedData.columns[selectedData]][i]*tempConfidence
        processedConfidence[processedConfidence.columns[selectedData]][i] += tempConfidence
    
    return processedData,processedConfidence

dataFirst = pd.read_csv("C:\\Users\\bcosk\\Desktop\\Tez_Kod\\ER12.csv", sep = ";")
dataTest = pd.read_csv("C:\\Users\\bcosk\\Desktop\\Tez_Kod\\ER13.csv", sep = ";")

now = datetime.now()


errorList1 = []
errorList2 = []
for selectedData in range(0,len(dataFirst.columns)):
    print("Current Selected Data: " + str(selectedData))
    inputData, outputData, outputData1 = dataPreparing(dataFirst,selectedData)
    inputTestData, outputTestData, outputTestData1 = dataPreparing(dataTest,selectedData)
    
    epochs = 1000
    validation_split = 0.2
    verbose = 1 # 0 silent, 1 progress, 2 text
    shuffle = True
    patience = 3
    mape = keras.losses.MeanAbsolutePercentageError(name='mape')
    es = EarlyStopping(monitor="val_mae", mode='auto', verbose=verbose, patience=patience, restore_best_weights=True)
        
    max_depth=10
    
    rf = DecisionTreeRegressor(criterion = "mae",random_state = 0,max_depth=max_depth)
    
    rf.fit(inputData, outputData)
    
    model1 = Sequential()
    model1.add(Dense(24,activation="relu",input_shape=(12,)))
    model1.add(Dense(1))
        
    model1.compile(optimizer="adam",loss = mape, metrics=[mape,"mae"])
    inputData1 = inputData.copy()
    inputData1["Confidence"] = outputData
    history = model1.fit(inputData1,outputData1,epochs=epochs,validation_split=validation_split, verbose = verbose, shuffle = shuffle, callbacks=[es])
    
    newColumnNames = []
    for i in range(0,len(dataTest.columns)):
        newColumnNames.append(dataTest.columns[i].replace("T","Q"))
        
    confidenceFirst = pd.DataFrame(np.zeros((len(dataTest),len(dataTest.columns))), columns = newColumnNames)
    processedDataFirst = dataTest.copy()
    processedConfidenceFirst = confidenceFirst.copy()
    
    confidenceErrorList = []
    dataErrorList = []
        
    # processedData,processedConfidence = randomNoiseGenerator(processedDataFirst,processedConfidenceFirst,selectedData) #for testing 2
    processedDataFirst,processedConfidenceFirst = randomNoiseGenerator(processedDataFirst,processedConfidenceFirst,selectedData) #for testing 3
    for i in range(0,len(dataTest.columns)):
        print("Testing " + str(i))
        # if i == selectedData: #for testing 2
        #     processedDataFirst = processedData
        #     processedConfidenceFirst = processedConfidence
        # else:
        #     processedDataFirst,processedConfidenceFirst = randomNoiseGenerator(processedData,processedConfidence,i)
        
        if i == selectedData: #for testing 3
            processedDataFirst = processedDataFirst
            processedConfidenceFirst = processedConfidenceFirst
        else:
            processedDataFirst,processedConfidenceFirst = randomNoiseGenerator(processedDataFirst,processedConfidenceFirst,i)
        
        confidencePredicted = rf.predict(processedDataFirst)
        
        tempProcessedData = processedDataFirst.copy()
        tempProcessedData["Confidence"] = confidencePredicted
        dataPredicted = pd.Series(model1.predict(tempProcessedData)[:,0])
        
        confidenceError = abs(confidencePredicted - processedConfidenceFirst[processedConfidenceFirst.columns[selectedData]])
        confidenceErrorList.append(sum(confidenceError)/len(confidenceError))
        
        dataError = abs((dataPredicted - dataTest[dataTest.columns[selectedData]])/dataTest[dataTest.columns[selectedData]])
        dataErrorList.append(sum(dataError)/len(dataError))
    
    errorList1.append(confidenceErrorList)
    errorList2.append(dataErrorList)

resultExcelName = now.strftime("%d_%m_%Y_%H_%M_%S") + "_AdvanceTesting3.xlsx"

writer = pd.ExcelWriter(resultExcelName, engine='xlsxwriter')
pd.DataFrame(errorList1).to_excel(writer, sheet_name='Noise Ratio Error List', index=False)
pd.DataFrame(errorList2).to_excel(writer, sheet_name='Data Error List', index=False)
writer.save()
