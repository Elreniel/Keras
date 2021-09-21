# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:23:47 2021

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

def constantNoiseGenerator(data,confidence,selectedData1,noise):
    processedData = data.copy()
    processedConfidence = confidence.copy()
    for i in range(41000,42000):
        tempConfidence = noise
        processedData[processedData.columns[selectedData1]][i] += processedData[processedData.columns[selectedData1]][i]*tempConfidence
        processedConfidence[processedConfidence.columns[selectedData1]][i] += tempConfidence
    
    return processedData,processedConfidence

def constantNoiseGenerator2(data,confidence,selectedData1,selectedData2,noise):
    processedData = data.copy()
    processedConfidence = confidence.copy()
    for i in range(41000,42000):
        tempConfidence = noise
        processedData[processedData.columns[selectedData1]][i] += processedData[processedData.columns[selectedData1]][i]*tempConfidence
        processedConfidence[processedConfidence.columns[selectedData1]][i] += tempConfidence
        
        processedData[processedData.columns[selectedData2]][i] += processedData[processedData.columns[selectedData2]][i]*tempConfidence
        processedConfidence[processedConfidence.columns[selectedData2]][i] += tempConfidence
    
    return processedData,processedConfidence

dataFirst = pd.read_csv("C:\\Users\\bcosk\\Desktop\\Tez_Kod\\ER12.csv", sep = ";")
dataTest = pd.read_csv("C:\\Users\\bcosk\\Desktop\\Tez_Kod\\ER13.csv", sep = ";")

now = datetime.now()

model1List = []
model2List = []

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
    
    model1List.append(rf)
    
    model1 = Sequential()
    model1.add(Dense(24,activation="relu",input_shape=(12,)))
    model1.add(Dense(1))
        
    model1.compile(optimizer="adam",loss = mape, metrics=[mape,"mae"])
    inputData1 = inputData.copy()
    inputData1["Confidence"] = outputData
    history = model1.fit(inputData1,outputData1,epochs=epochs,validation_split=validation_split, verbose = verbose, shuffle = shuffle, callbacks=[es])
    
    model2List.append(model1)

newColumnNames = []
for i in range(0,len(dataTest.columns)):
    newColumnNames.append(dataTest.columns[i].replace("T","Q"))
    
confidenceFirst = pd.DataFrame(np.zeros((len(dataTest),len(dataTest.columns))), columns = newColumnNames)
processedDataFirst = dataTest.copy()
processedConfidenceFirst = confidenceFirst.copy()

resultExcelName = now.strftime("%d_%m_%Y_%H_%M_%S") + "_AdvanceTesting_2.xlsx"

writer = pd.ExcelWriter(resultExcelName, engine='xlsxwriter')

constantNoiseList = [-0.75,-0.5,-0.25,0.25,0.5,0.75]
dataPairList = [0,2,4,7,8,10]
for curIndex in range(0,len(constantNoiseList)):
    
    confidencePredictedList = []
    dataPredictedList = []
    
    constantNoise = constantNoiseList[curIndex]
    # dataPair1 = dataPairList[2*curIndex]
    # dataPair2 = dataPairList[2*curIndex + 1]
    selectedData = dataPairList[curIndex]
    
    processedData,processedConfidence = constantNoiseGenerator(processedDataFirst,processedConfidenceFirst,selectedData,constantNoise)
    
    processedData = processedData[40000:43000]
    processedConfidence = processedConfidence[40000:43000]
    
    for i in range(0,len(dataTest.columns)):
        print("Testing " + str(i))
        
        confidencePredicted = model1List[i].predict(processedData)
        
        tempProcessedData = processedData.copy()
        tempProcessedData["Confidence"] = confidencePredicted
        dataPredicted = pd.Series(model2List[i].predict(tempProcessedData)[:,0])
        
        confidencePredictedList.append(confidencePredicted)
        if i == selectedData:
            dataPredictedList.append(pd.Series(dataTest[dataTest.columns[selectedData]].loc[40000:42999].array._ndarray))
            dataPredictedList.append(dataPredicted)
    
    sheetName1 = "NoiseRatioResults_" + str(curIndex)
    sheetName2 = "DataResults_" + str(curIndex)
    pd.DataFrame(confidencePredictedList).to_excel(writer, sheet_name=sheetName1, index=False)
    pd.DataFrame(dataPredictedList).to_excel(writer, sheet_name=sheetName2, index=False)
writer.save()
