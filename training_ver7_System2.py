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
errorList2 = []

now = datetime.now()

for selectedData in range(0,len(dataFirst.columns)):
    inputData, outputData, outputData1 = dataPreparing(dataFirst,selectedData)
    inputTestData, outputTestData, outputTestData1 = dataPreparing(dataTest,selectedData)
    
    epochs = 1000
    validation_split = 0.2
    verbose = 1 # 0 silent, 1 progress, 2 text
    shuffle = True
    patience = 3
    mape = keras.losses.MeanAbsolutePercentageError(name='mape')
    es = EarlyStopping(monitor='loss', mode='auto', verbose=verbose, patience=patience, restore_best_weights=True)

    max_depth=10
    
    model = DecisionTreeRegressor(criterion = "mae",random_state = 0,max_depth=max_depth)
    
    model.fit(inputData, outputData)
    
    
    # model1 = Sequential()
    # model1.add(Dense(24,activation="relu",input_shape=(12,)))
    # model1.add(Dense(1))
        
    # model1.compile(optimizer="adam",loss = mape, metrics=[mape,"mae"])
    # inputData1 = inputData.copy()
    # inputData1["Confidence"] = outputData
    # history = model1.fit(inputData1,outputData1,epochs=epochs,validation_split=validation_split, verbose = verbose, shuffle = shuffle, callbacks=[es])
    
    # inputTestData1 = inputTestData.copy()
    # inputTestData1["Confidence"] = outputTestData
    # testResult = model1.evaluate(inputTestData1,outputTestData1,verbose=verbose)
    
    # model2 = DecisionTreeRegressor(criterion = "mae",random_state = 0,max_depth=max_depth)
    
    # inputData1 = inputData[inputData.columns[selectedData]].copy().to_frame().join(outputData)
    # model2.fit(inputData1,outputData1)

    # if selectedData == 8:
    #     newColumnNames = []
    #     for i in range(0,len(dataTest.columns)):
    #         newColumnNames.append(dataTest.columns[i].replace("T","Q"))
            
    #     confidenceFirst = pd.DataFrame(np.zeros((len(dataTest),len(dataTest.columns))), columns = newColumnNames)
    #     processedDataFirst = dataTest.copy()
    #     processedconfidenceFirst = confidenceFirst.copy()
    #     tempConfidence = 0.5
                    
    #     for i in range(int(len(dataTest)/2),len(dataTest)):
    #         processedDataFirst[processedDataFirst.columns[selectedData]][i] += processedDataFirst[processedDataFirst.columns[selectedData]][i] * tempConfidence
    #         processedconfidenceFirst[processedconfidenceFirst.columns[selectedData]][i] += tempConfidence
        
    #     dataPredicted = dataTest.copy()
    #     confidencePredicted = confidenceFirst.copy()
    #     confidencePredicted[confidencePredicted.columns[selectedData]] = model.predict(processedDataFirst)
    #     dataPredicted[dataPredicted.columns[selectedData]] = processedDataFirst[processedDataFirst.columns[selectedData]] - (processedDataFirst[processedDataFirst.columns[selectedData]] * confidencePredicted[confidencePredicted.columns[selectedData]])
        
    #     dataPredicted1 = dataTest.copy()
    #     tempProcessedDataFirst = processedDataFirst.copy()
    #     tempProcessedDataFirst["Confidence"] = confidencePredicted[confidencePredicted.columns[selectedData]]
    #     dataPredicted1[dataPredicted1.columns[selectedData]] = model1.predict(tempProcessedDataFirst)
        
    #     dataPredicted2 = dataTest.copy()
    #     processedDataFirst1 = processedDataFirst[processedDataFirst.columns[selectedData]].copy().to_frame().join(confidencePredicted[confidencePredicted.columns[selectedData]])
    #     dataPredicted2[dataPredicted2.columns[selectedData]] = model2.predict(processedDataFirst1)
                
    #     rangeMin = 40000    
    #     rangeMax = 80000
        
    #     ax = dataTest.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataTest.columns[selectedData],x="index",s=0.05,c="red")
    #     dataPredicted.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataPredicted.columns[selectedData],ax = ax,x="index",s=0.05,c="blue")
    #     plt.legend(["T9","Predicted T9 by Linear Model"])
    #     plt.grid()
    #     plt.xlabel("Timestamp")
    #     plt.ylabel("Temperature")
    #     plt.savefig(now.strftime("%d_%m_%Y_%H_%M_%S") + "System2_1", dpi = 300)
    #     plt.show()
        
    #     ax = dataTest.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataTest.columns[selectedData],x="index",s=0.05,c="red")
    #     dataPredicted1.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataPredicted1.columns[selectedData],ax = ax,x="index",s=0.05,c="blue")
    #     plt.legend(["T9","Predicted T9 by Neural Network"])
    #     plt.grid()
    #     plt.xlabel("Timestamp")
    #     plt.ylabel("Temperature")
    #     plt.savefig(now.strftime("%d_%m_%Y_%H_%M_%S") + "System2_2", dpi = 300)
    #     plt.show()
        
    #     ax = dataTest.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataTest.columns[selectedData],x="index",s=0.05,c="red")
    #     dataPredicted2.reset_index().iloc[rangeMin:rangeMax].plot.scatter(y = dataPredicted2.columns[selectedData],ax = ax,x="index",s=0.05,c="blue")
    #     plt.legend(["T9","Predicted T9 by Decision Tree"])
    #     plt.grid()
    #     plt.xlabel("Timestamp")
    #     plt.ylabel("Temperature")
    #     plt.savefig(now.strftime("%d_%m_%Y_%H_%M_%S") + "System2_3", dpi = 300)
    #     plt.show()

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
        dataPredicted[dataPredicted.columns[selectedData]] = processedDataFirst[processedDataFirst.columns[selectedData]] / (1 + confidencePredicted[confidencePredicted.columns[selectedData]])
        
        # dataPredicted1 = dataTest.copy()
        # tempProcessedDataFirst = processedDataFirst.copy()
        # tempProcessedDataFirst["Confidence"] = confidencePredicted[confidencePredicted.columns[selectedData]]
        # dataPredicted1[dataPredicted1.columns[selectedData]] = model1.predict(tempProcessedDataFirst)
        
        # dataPredicted2 = dataTest.copy()
        # processedDataFirst1 = processedDataFirst[processedDataFirst.columns[selectedData]].copy().to_frame().join(confidencePredicted[confidencePredicted.columns[selectedData]])
        # dataPredicted2[dataPredicted2.columns[selectedData]] = model2.predict(processedDataFirst1)
                
        errorList = 100 * abs((dataPredicted[dataPredicted.columns[selectedData]] - dataTest[dataTest.columns[selectedData]]) / dataTest[dataTest.columns[selectedData]])
        tempErrorList0.append(sum(errorList)/len(errorList))

        # errorList = 100 * abs((dataPredicted1[dataPredicted1.columns[selectedData]] - dataTest[dataTest.columns[selectedData]]) / dataTest[dataTest.columns[selectedData]])
        # tempErrorList1.append(sum(errorList)/len(errorList))

        # errorList = 100 * abs((dataPredicted2[dataPredicted2.columns[selectedData]] - dataTest[dataTest.columns[selectedData]]) / dataTest[dataTest.columns[selectedData]])
        # tempErrorList2.append(sum(errorList)/len(errorList))
    errorList0.append(tempErrorList0)
    # errorList1.append(tempErrorList1)
    # errorList2.append(tempErrorList2)

dfErrorList0 = pd.DataFrame(errorList0)
# dfErrorList1 = pd.DataFrame(errorList1)
# dfErrorList2 = pd.DataFrame(errorList2)

resultExcelName = now.strftime("%d_%m_%Y_%H_%M_%S") + "_System2_Testing.xlsx"

writer = pd.ExcelWriter(resultExcelName, engine='xlsxwriter')
dfErrorList0.to_excel(writer, sheet_name='Linear Model', index=False)
# dfErrorList1.to_excel(writer, sheet_name='Neural Network', index=False)
# dfErrorList2.to_excel(writer, sheet_name='Decision Tree', index=False)
writer.save()
