# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:48:04 2021

@author: evinao
"""
from pathlib import Path
import pandas as pd
import Tools.signal_analysis_tools as sat
import Tools.signal_visualization_tool as svt
import numpy as np
import Tools.feturesCorrelations_tools as fct
import matplotlib.pyplot as plt


factorList = ['AE','RE', 'AA', 'PI'] #, 
contentList = ['C1', 'C2', 'C3', 'C4'] #
isOnlyLowHighScoreUsersList = [False] #True,

#copy settingss

feature_pars = []
feature_function = ''


saveStdMeanKurtosisQ = False

# not copied part
for selectedFactor in factorList:
    outputFolder = outputFolderBaseName +selectedFactor + "/"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    pValuesFileName = outputFolder + '/AllPValues.csv'  #+ selectedContent
    
    for contentID in contentList:
        contentUserAdStartEndTimesFile = rootFolder + "MetaData/" + contentID + "_usersAdStartEndTimes.csv"
        for feature_code in feature_codes:
            for isOnlyLowHighScoreUsers in isOnlyLowHighScoreUsersList:
                for index in range(6):
                                            
                    #save the std mean kurtosis only once for each signal
                    saveStdMeanKurtosisQ = True
                    if index==0:
                        #avaraged diameter
                        signalName = 'diameter'
                        sensors = {1:['tobii', {'diameter':['diameter']}]}
                        sensorsFeaturePars = {1:['tobii', {'diameter':[0, 1.6, 4]}]}
                        cut_f = 0.27 # Hertz              
                                 
                    elif index==1:
                        signalName = 'EDA'
                        sensors = {1:['empatica', {'EDA':['EDA']}]}
                        sensorsFeaturePars = {1:['empatica', {'EDA':[0, 0.015]}]} # 0.02
                        cut_f = 0.12 # Hertz
                    elif index==2:
                        signalName = 'HR'
                        sensors = {1:['empatica', {'HR':['HR']}]}
                        sensorsFeaturePars = {1:['empatica', {'HR':[0, 0.06, 0.12, 0.20]}]} #[0, 0.06, 0.12, 0.20] #0, 0.015, 0.030, 0.045
                        cut_f = 1.8 # Hertz
                    elif index==3:
                        signalName = 'AccX'
                        sensors = {1:['empatica', {'ACC':['AccX']}]}
                        sensorsFeaturePars = {1:['empatica', {'AccX':[0, 0.075, 0.147, 0.223, 0.280]}]}
                        cut_f = 0.3 # Hertz
                    elif index==4:
                        signalName = 'GSR_Skin_Conductance_microSiemens'
                        sensors = {1:['shimmer', {'':['GSR_Skin_Conductance_microSiemens']}]}
                        sensorsFeaturePars = {1:['shimmer', {'GSR_Skin_Conductance_microSiemens':[0, 0.015, 0.045, 0.075]}]} #[0, 0.015, 0.045, 0.060, 0.075]
                        cut_f = 0.12 # Hertz
                    elif index==5:
                        signalName = 'Temperature_BMP280_Degrees Celsius'
                        sensors = {1:['shimmer', {'':['Temperature_BMP280_Degrees Celsius']}]}
                        sensorsFeaturePars = {1:['shimmer', {'Temperature_BMP280_Degrees Celsius':[0, 0.075, 0.147, 0.223, 0.280]}]}
                        cut_f = 0.02 # Hertz
                        
                    if feature_code == 'spec_phs' or feature_code == 'spec_pow':
                        feature_pars = sensorsFeaturePars[sensorID][1][signalName]                        
                        feature_function = 'getF1oF23'
                    else:
                        feature_pars = [cut_f]
                        feature_function = ''
                        
                        
                        #copy allt he things from the corr_vsigs
                        
                        
                        
                        
                        plt.close('all')