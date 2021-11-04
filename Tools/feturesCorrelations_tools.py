# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:45:26 2021

@author: Andrej Košir
"""
import numpy as np
import pandas as pd
from pathlib import Path
import Tools.signal_analysis_tools as sat

#%% Functions

#int he file uID is userID, rename it to be uID to fit better with the rest of the script
# def readFactorScoresFile(factorScoresFileName):
#     scores_df = pd.read_csv(factorScoresFileName)
#     scores_df.rename({'userID': 'uID'}, axis=1, inplace=True)
    
#     return scores_df

def getAllUsersScores(userIDList, factorScoresFileName, selectedFactor):
    scores_df = pd.read_csv(factorScoresFileName) #readFactorScoresFile(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['uID'].isin(userIDList)]
    
    return scores_df[['uID', selectedFactor]]

def getAllFilteredUserScores(outputFolder, selectedFactor, contentID, sensors, sensorID, feature_code, userSensorContentFileName, factorScoresFileName):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0] + '/' + feature_code).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    allUsers = getAllUsersScores(filteredUserList, factorScoresFileName, selectedFactor)
    return allUsers
    
def getLowAndHighFactorUserIDS(outputFolder, userIDList, contentID, sensors, sensorID, feature_code, factorScoresFileName, selectedFactor, numberOfUsers):
    Path(outputFolder + contentID + '/' + sensors[sensorID][0] + '/' + feature_code).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    scores_df = pd.read_csv(factorScoresFileName) #readFactorScoresFile(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['uID'].isin(userIDList)] 
    sorted_df = scores_df.sort_values(selectedFactor)
    
    return sorted_df[['uID', selectedFactor]][:numberOfUsers], sorted_df[['uID', selectedFactor]][-numberOfUsers:]
    

# print(fileNameGenerator(1, 2))

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df['uID']


def getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, contentID, sensors, sensorID, feature_code, userSensorContentFileName, factorScoresFileName, numberOfUsers):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(outputFolder, filteredUserList, contentID, sensors, sensorID, feature_code, factorScoresFileName, selectedFactor, numberOfUsers)
    
    return lowFactorUserIDs, highFactorUserIDs
        

def fileNameGenerator(rootFolder, higherSamplingFreqAndFolderExtension, userID, sensors, sensorID, sensorFileNameExt = "", readFromHigherResolutionSignalQ = False):
    if readFromHigherResolutionSignalQ:
        fileNameStr = rootFolder + str(userID) + "/Resampled_" + str(higherSamplingFreqAndFolderExtension) +"/uID-" + str(userID) + "_" + sensors[sensorID][0]
    else:
        fileNameStr = rootFolder + str(userID) + "/Resampled/uID-" + str(userID) + "_" + sensors[sensorID][0]
    if sensorFileNameExt:
        fileNameStr = fileNameStr + '_'  + sensorFileNameExt + '_resampled.csv'
    else:
        fileNameStr = fileNameStr + '_resampled.csv'
        
    return fileNameStr
    
def readSignal(fileName, signalName):
    df = pd.read_csv(fileName)
    
    return df[['timestamp_s',signalName]]

def getSignalDf(rootFolder, higherSamplingFreqAndFolderExtension, userID, sensors, sensorID, signalName, sensorFileNameExt = ""):
    fileName = fileNameGenerator(rootFolder, higherSamplingFreqAndFolderExtension, userID, sensors, sensorID, sensorFileNameExt)
    return readSignal(fileName, signalName)

def writeFeatureToDf(users, outputFolder, selectedContent, sensors, sensorID, signalName, feature_code, selectedFactor, isOnlyLowHighScoreUsers, cutSignals):
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/FeatureVals_"  + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code 
    
    if isOnlyLowHighScoreUsers:
        fName = fName + ' low-high_users'
    else:
        fName = fName + ' all_users'
    
    if cutSignals:
        fName = fName + '_cutSignal'
    else:
        fName = fName + '_allSignal'
        
    fName = fName+ '_df.csv' 
    
    # user_df = pd.DataFrame(users.items(), columns=['uID', feature_code, selectedFactor])
    
    userList, featureValList, selectedFactorScores = [k for k,v in users.items()], [v[sensorID][signalName][feature_code][0] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()]  # userID 
    
    user_arr = np.transpose(np.array([userList, featureValList, selectedFactorScores]))
    user_df = pd.DataFrame(data= user_arr, columns=['uID', feature_code, selectedFactor])
    
    user_df.to_csv(fName, index=False)
    
def readFeatureDf_FromFileName(fileName):
    return pd.read_csv(fileName)
    
def readFeatureDf(outputFolder, selectedContent, sensors, sensorID, signalName, feature_code, isOnlyLowHighScoreUsers, cutSignals):
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/FeatureVals_"  + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code 
    
    if isOnlyLowHighScoreUsers:
        fName = fName + ' low-high_users'
    else:
        fName = fName + ' all_users'
    
    if cutSignals:
        fName = fName + '_cutSignal'
    else:
        fName = fName + '_allSignal'
        
    fName = fName+ '_df.csv'
    
    return pd.read_csv(fName)

def saveNumpyArraysToDf(sig_t, sig_x, sig_p_x, time_int, selectedContent, cuID, sensors, sensorID, signalName, cutdfFolder):
    cut_sig_t, cut_sig_x = sat.getCutSignal(sig_t, sig_x, time_int)
    cut_sig_t_df = pd.DataFrame(data=cut_sig_t, columns=['timestamp_s'])
    cut_sig_x_df = pd.DataFrame(data=cut_sig_x, columns=[signalName])
    cut_df = pd.concat([cut_sig_t_df,cut_sig_x_df], axis=1)
    cutdfFolderPath = cutdfFolder + selectedContent + '/' + sensors[sensorID][0] +'/'
    Path(cutdfFolderPath).mkdir(parents=True, exist_ok=True)
    cut_df.to_csv(cutdfFolderPath + 'uID_' + str(cuID) + '_' + sensors[sensorID][0] + '_' + signalName + '_' + selectedContent + '.csv')
    
    cut_sig_t, cut_sig_p_x = sat.getCutSignal(sig_t, sig_p_x, time_int)
    cut_sig_p_x_df = pd.DataFrame(data=cut_sig_p_x, columns=[signalName])
    cut_df_prep = pd.concat([cut_sig_t_df,cut_sig_p_x_df], axis=1)
    cut_df_prep.to_csv(cutdfFolderPath + 'uID_' + str(cuID) + '_' + sensors[sensorID][0] + '_' + signalName + '_' + selectedContent + '_preprocessed.csv')
 
      
