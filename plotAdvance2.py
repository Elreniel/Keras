# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:54:16 2021

@author: bcosk
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

dataList = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11"]

resultsNoiseRatioList = []
resultsDataList = []

excelName = "C:/Users/bcosk/Desktop/Tez_Kod/24_04_2021_16_42_35_AdvanceTesting_2.xlsx"

constantNoiseList = [-0.75,-0.5,-0.25,0.25,0.5,0.75]
dataPairList = [0,2,4,7,8,10]

for i in range(0,len(constantNoiseList)):   
    sheetNameNoiseRatio = "NoiseRatioResults_" + str(i)
    sheetNameData = "DataResults_" + str(i)
    
    resultsNoiseRatio = pd.read_excel(excelName,sheet_name=sheetNameNoiseRatio).T
    resultsData = pd.read_excel(excelName,sheet_name=sheetNameData).T
    
    resultsNoiseRatio.columns = dataList
    resultsData.columns = [dataList[dataPairList[i]],dataList[dataPairList[i]] + str("_Predicted")]
    
    resultsData[dataList[dataPairList[i]] + str("'")] = resultsData[dataList[dataPairList[i]]]
    for k in range(1000,2000):
        resultsData[dataList[dataPairList[i]] + str("'")][k] = resultsData[dataList[dataPairList[i]]][k] + (resultsData[dataList[dataPairList[i]]][k] * constantNoiseList[i]) 
        
    resultsNoiseRatio["Index"] = np.arange(40000,43000,1)
    resultsData["Index"] = np.arange(40000,43000,1)
    
    resultsNoiseRatioList.append(resultsNoiseRatio)
    resultsDataList.append(resultsData)

for selectedData in range(0,len(constantNoiseList)):
    # resultsNoiseRatioList[selectedData].plot(x = "Index")
    # plt.grid()
    # plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
    # plt.xlabel("Time")
    # plt.ylabel("Noise Ratio")
    # plt.ylim([-1,1])
    # tempTitle = "Scenario " + str(selectedData + 1) + " - Predicted Noise Ratio \n " + dataList[dataPairList[selectedData]] + " to " + str(constantNoiseList[selectedData])
    # plt.title(tempTitle)
    # rect = patches.Rectangle(xy=(40000, -0.25), width=3000, height=0.5, alpha = 0.5, facecolor = "gray")
    # plt.gca().add_patch(rect)
    # plt.show()
    # plt.savefig("Scenario " + str(selectedData + 1), dpi = 300)
    
    resultsDataList[selectedData].plot(x = "Index")
    plt.grid()
    # plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.ylim([0,120])
    tempTitle = "Scenario " + str(selectedData + 1) + " - Predicted Data \n " + dataList[dataPairList[selectedData]] + " to " + str(constantNoiseList[selectedData])
    plt.title(tempTitle)
    # plt.show()
    plt.savefig("Scenario_Data " + str(selectedData + 1), dpi = 300)

    
# data = pd.read_csv("C:/Users/bcosk/Desktop/Tez_Kod/ER12.csv", sep = ";")
# columns = data.columns.values
# data.plot()
# plt.grid()
# plt.xlabel("Time")
# plt.ylabel("Data")
# plt.ylim([0,120])
# plt.title("Real life testing scenarios distorted part")
# rect = patches.Rectangle(xy=(40000, 0), width=3000, height=120, alpha = 0.5, facecolor = "black")
# plt.gca().add_patch(rect)
# plt.show()
# plt.savefig("AllDataRect", dpi = 300)