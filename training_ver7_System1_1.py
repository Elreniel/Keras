# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:31:23 2020
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

dataFirst = pd.read_csv("P:\Tez\Data\ER12.csv", sep = ";")
dataTest = pd.read_csv("P:\Tez\Data\ER13.csv", sep = ";")
errorList0 = []
errorList1 = []

now = datetime.now()

for selectedData in range(0,len(dataTest.columns)):
# for selectedData in range(9,10):
    inputData, outputData, outputData1 = dataPreparing(dataFirst,selectedData)
    inputTestData, outputTestData, outputTestData1 = dataPreparing(dataTest,selectedData)
    
    epochs = 1000
    validation_split = 0.2
    verbose = 1 # 0 silent, 1 progress, 2 text
    shuffle = True
    patience = 3
    mape = keras.losses.MeanAbsolutePercentageError(name='mape')
    es = EarlyStopping(monitor="val_mae", mode='auto', verbose=verbose, patience=patience, restore_best_weights=True)

    model = Sequential()
    model.add(Dense(22,activation="relu",input_shape=(11,)))
    model.add(Dense(1))
    
    model.compile(optimizer="adam",loss = "mae", metrics=["mae"])
    history = model.fit(inputData,outputData,epochs=epochs,validation_split=validation_split, verbose = verbose, shuffle = shuffle, callbacks=[es])
    testResult = model.evaluate(inputTestData,outputTestData,verbose=verbose)
        
    max_depth=10
    
    rf = DecisionTreeRegressor(criterion = "mae",random_state = 0,max_depth=max_depth)
    
    rf.fit(inputData, outputData)

    if selectedData == 8:
        newColumnNames = []
        for i in range(0,len(dataTest.columns)):
            newColumnNames.append(dataTest.columns[i].replace("T","Q"))
            
        confidenceFirst = pd.DataFrame(np.zeros((len(dataTest),len(dataTest.columns))), columns = newColumnNames)
        processedDataFirst = dataTest.copy()
        processedconfidenceFirst = confidenceFirst.copy()
        tempConfidence = 0.5
                    
        for i in range(int(len(dataTest)/2),len(dataTest)):
            processedDataFirst[processedDataFirst.columns[selectedData]][i] += processedDataFirst[processedDataFirst.columns[selectedData]][i] * tempConfidence
            processedconfidenceFirst[processedconfidenceFirst.columns[selectedData]][i] += tempConfidence
        
        dataPredicted = dataTest.copy()
        confidencePredicted = confidenceFirst.copy()
        confidencePredicted[confidencePredicted.columns[selectedData]] = model.predict(processedDataFirst)
        
        confidencePredicted1 = confidenceFirst.copy()
        confidencePredicted1[confidencePredicted1.columns[selectedData]] = rf.predict(processedDataFirst)
                
        rangeMin = 40000    
        rangeMax = 80000
        ax = processedconfidenceFirst.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = processedconfidenceFirst.columns[selectedData],x="index",s=0.05,c="red")
        confidencePredicted.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = confidencePredicted.columns[selectedData],x="index",s=0.05,c="blue",ax=ax)
        plt.legend(["v","Predicted v by Neural Network"])
        plt.grid()
        plt.xlabel("Timestamp")
        plt.ylabel("Noise Ratio")
        plt.savefig(now.strftime("%d_%m_%Y_%H_%M_%S") + "System1_1", dpi = 300)
        plt.show()
        
        ax = processedconfidenceFirst.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = processedconfidenceFirst.columns[selectedData],x="index",s=0.05,c="red")
        confidencePredicted1.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = confidencePredicted1.columns[selectedData],x="index",s=0.05,c="blue",ax=ax)
        plt.legend(["v","Predicted v by Decision Tree"])
        plt.grid()
        plt.xlabel("Timestamp")
        plt.ylabel("Noise Ratio")
        plt.savefig(now.strftime("%d_%m_%Y_%H_%M_%S") + "System1_2", dpi = 300)
        plt.show()

    print("Testing: " + str(selectedData))
    tempErrorList0 = []
    tempErrorList1 = []
    tempErrorList2 = []
    for k in range(-10,11):
        newColumnNames = []
        for i in range(0,len(dataTest.columns)):
            newColumnNames.append(dataTest.columns[i].replace("T","Q"))
            
        confidenceFirst = pd.DataFrame(np.zeros((len(dataTest),len(dataTest.columns))), columns = newColumnNames)
        processedDataFirst = dataTest.copy()
        processedconfidenceFirst = confidenceFirst.copy()
        tempConfidence = k/10

        for i in range(0,len(dataTest)):
            processedDataFirst[processedDataFirst.columns[selectedData]][i] += processedDataFirst[processedDataFirst.columns[selectedData]][i] * tempConfidence
            processedconfidenceFirst[processedconfidenceFirst.columns[selectedData]][i] += tempConfidence
        
        dataPredicted = dataTest.copy()
        confidencePredicted = confidenceFirst.copy()
        confidencePredicted[confidencePredicted.columns[selectedData]] = model.predict(processedDataFirst)
        # dataPredicted[dataPredicted.columns[selectedData]] = processedDataFirst[processedDataFirst.columns[selectedData]] - (processedDataFirst[processedDataFirst.columns[selectedData]] * confidencePredicted[confidencePredicted.columns[selectedData]])
        
        dataPredicted1 = dataTest.copy()
        confidencePredicted1 = confidenceFirst.copy()
        confidencePredicted1[confidencePredicted1.columns[selectedData]] = rf.predict(processedDataFirst)
        # dataPredicted1[dataPredicted1.columns[selectedData]] = processedDataFirst[processedDataFirst.columns[selectedData]] - (processedDataFirst[processedDataFirst.columns[selectedData]] * confidencePredicted1[confidencePredicted1.columns[selectedData]])
                       
        errorList = abs((confidencePredicted[confidencePredicted.columns[selectedData]] - processedconfidenceFirst[processedconfidenceFirst.columns[selectedData]]))
        tempErrorList0.append(sum(errorList)/len(errorList))

        errorList = abs((confidencePredicted1[confidencePredicted1.columns[selectedData]] - processedconfidenceFirst[processedconfidenceFirst.columns[selectedData]]))
        tempErrorList1.append(sum(errorList)/len(errorList))

    errorList0.append(tempErrorList0)
    errorList1.append(tempErrorList1)
    
dfErrorList0 = pd.DataFrame(errorList0)
dfErrorList1 = pd.DataFrame(errorList1)

resultExcelName = now.strftime("%d_%m_%Y_%H_%M_%S") + "_System1_Testing.xlsx"

writer = pd.ExcelWriter(resultExcelName, engine='xlsxwriter')
dfErrorList0.to_excel(writer, sheet_name='NeuralNetwork', index=False)
dfErrorList1.to_excel(writer, sheet_name='DecisionTree', index=False)
writer.save()
