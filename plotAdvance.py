# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:31:15 2021

@author: bcosk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataList = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11"]

resultsNoiseRatio = pd.read_excel("C:/Users/bcosk/Desktop/Tez_Kod/17_04_2021_23_17_23_AdvanceTesting2.xlsx",sheet_name="Noise Ratio Error List").T
resultsData = pd.read_excel("C:/Users/bcosk/Desktop/Tez_Kod/17_04_2021_23_17_23_AdvanceTesting2.xlsx",sheet_name="Data Error List").T
resultsNoiseRatio["Index"] = dataList
resultsData["Index"] = dataList

resultsNoiseRatio.plot(x = "Index")
plt.grid()
plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
plt.xlabel("Data")
plt.ylabel("MAE")
# plt.ylim([0,0.5])
plt.title("Noise Ratio Error")
# plt.xlabel(dataList)
plt.xticks(np.arange(0,len(dataList),1),dataList)
# plt.show()
plt.savefig("Advance11", dpi = 300)

resultsData.plot(x = "Index")
plt.grid()
plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
plt.xlabel("Data")
plt.ylabel("MAPE")
# plt.ylim([0,0.5])
plt.title("Data Prediction Error")
plt.xticks(np.arange(0,len(dataList),1),dataList)
# plt.show()
plt.savefig("Advance12", dpi = 300)

resultsNoiseRatio1 = pd.read_excel("C:/Users/bcosk/Desktop/Tez_Kod/18_04_2021_15_10_34_AdvanceTesting3.xlsx",sheet_name="Noise Ratio Error List").T
resultsData1 = pd.read_excel("C:/Users/bcosk/Desktop/Tez_Kod/18_04_2021_15_10_34_AdvanceTesting3.xlsx",sheet_name="Data Error List").T
resultsNoiseRatio1["Index"] = dataList
resultsData1["Index"] = dataList

resultsNoiseRatio1.plot(x = "Index")
plt.grid()
plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
plt.xlabel("Data")
plt.ylabel("MAE")
# plt.ylim([0,0.5])
plt.title("Noise Ratio Error")
plt.xticks(np.arange(0,len(dataList),1),dataList)
plt.show()
# plt.savefig("Advance21", dpi = 300)

resultsData1.plot(x = "Index")
plt.grid()
plt.legend(dataList,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol = 6)
plt.xlabel("Data")
plt.ylabel("MAPE")
# plt.ylim([0,0.5])
plt.title("Data Prediction Error")
plt.xticks(np.arange(0,len(dataList),1),dataList)
plt.show()
# plt.savefig("Advance22", dpi = 300)