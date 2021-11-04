# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 07:50:09 2021

@author: Andrej Košir
"""

# @brief Functions for pictorial examination of sensor data


import matplotlib.pyplot as plt
import pandas as pd
from . import Utils
from datetime import datetime
import os.path
from pathlib import Path




def fileNameGenerator(rootFolder, sensors, userID, sensorID, sensorFileNameExt = ""):
    fileNameStr = rootFolder + str(userID) + "/Resampled/uID-" + str(userID) + "_" + sensors[sensorID][0]
    if sensorFileNameExt:
        fileNameStr = fileNameStr + '_'  + sensorFileNameExt + '_resampled.csv'
    else:
        fileNameStr = fileNameStr + '_resampled.csv'
        
    return fileNameStr
    
def readSignal(fileName, signalName):
    df = pd.read_csv(fileName)
    
    return df[['timestamp_s',signalName]]

def getSignalDf(rootFolder, sensors, userID, sensorID, signalName, sensorFileNameExt = ""):
    fileName = fileNameGenerator(rootFolder, sensors, userID, sensorID, sensorFileNameExt)
    return readSignal(fileName, signalName)


def generateSignalSubplots(rootFolder, sensors, ax, axIndex, lowOrHighFactorUserIDs, sensorID, signalName, sensorFileNameExt):
    subPlotId = 0    
    for userID in lowOrHighFactorUserIDs:
        # if (key % 2) == 0:
        #     color = '#1f77b4'
        # else:
        #     color =  'm'
          
        df = getSignalDf(rootFolder, sensors, userID, sensorID, signalName, sensorFileNameExt)
        print(df.head())
        ax[axIndex][subPlotId].plot(df['timestamp_s'], df[signalName]) #, c=color
        ax[axIndex][subPlotId].set_xlabel('t [s]')
        ax[axIndex][subPlotId].set_ylabel('user ' + str(userID))
        ax[axIndex][subPlotId].grid(True)
        subPlotId = subPlotId + 1

def generateAdVidLines(ax, axIndex, usersDictFileName,lowOrHighFactorUserIDs,selectedContent):
    
    out_times_lst = Utils.get_video_and_ad_times_userslist_one_content(usersDictFileName,
                                                                       lowOrHighFactorUserIDs,
                                                                       selectedContent)
    subPlotId = 0  
    #draw vertical lines for one of the video-ad 
    for ad in out_times_lst:
        ax[axIndex][subPlotId].axvline(x=ad[0], c='g')
        ax[axIndex][subPlotId].axvline(x=ad[1], c='r')
        ax[axIndex][subPlotId].axvline(x=ad[2], c='r')
        ax[axIndex][subPlotId].axvline(x=ad[3], c='g')
        subPlotId = subPlotId + 1

def generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(rootFolder, sensors, contentID, selectedFactor, usersDictFileName, outputFolderName,
                                                               lowFactorUserIDs,
                                                               highFactorUserIDs,
                                                               sensorID,
                                                               signalName,
                                                               sensorFileNameExt = ""):
    
    subPlotCount = max(len(lowFactorUserIDs), len(highFactorUserIDs))
    #take min and max 
    fig, ax = plt.subplots(2, subPlotCount) #, sharey=True
    fig.suptitle(sensors[sensorID][0] +' ' + sensorFileNameExt +' ' + signalName + ' ' + contentID)
    
    ax[0][0].set_title('Low score ' + selectedFactor)
    ax[1][0].set_title('High score ' + selectedFactor)
    
    # plot the first row of subplots
    generateSignalSubplots(rootFolder, sensors, ax, 0, lowFactorUserIDs, sensorID, signalName, sensorFileNameExt)    

    # plot the second row of subplots
    generateSignalSubplots(rootFolder, sensors, ax, 1, highFactorUserIDs, sensorID, signalName, sensorFileNameExt)
    
    # draw the first row of video ad lines
    generateAdVidLines(ax, 0, usersDictFileName, lowFactorUserIDs, contentID)    
    # draw the second row of video ad lines
    generateAdVidLines(ax, 1, usersDictFileName, highFactorUserIDs, contentID) 
    
    now = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    if '/' in signalName:
        signalName = signalName.replace('/','-')
    elif '.' in signalName:
        signalName = signalName.replace('.','-')
    saveFileFig = outputFolderName + contentID + '/' + sensors[sensorID][0] + '/' + selectedFactor + '_' + sensors[sensorID][0] + '_' + sensorFileNameExt + '_' + signalName + '_' + now
    plt.savefig(saveFileFig +'.jpg')
    Utils.writePickleFile(fig, saveFileFig)
    # pickle.dump(fig, open(saveFileFig +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
    plt.close(fig)
    # plt.show()
    
    


def getLowAndHighFactorUserIDS(userIDList, factorScoresFileName, selectedFactor, numberOfUsers=4):
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['uID'].isin(userIDList)] 
    sorted_df = scores_df.sort_values(selectedFactor)
    
    # print(sorted_df['userID'][:numberOfUsers])
    # print(sorted_df['userID'][-numberOfUsers:])
    return sorted_df['uID'][:numberOfUsers], sorted_df['uID'][-numberOfUsers:]
    

# print(fileNameGenerator(1, 2))

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID, userIDColumnName):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df[userIDColumnName]





def createFiguresForAll(rootFolder, pictOutputFolder, contentID, selectedFactor, usersDictFileName, outputFolderName, userIDColumnName, sensors, userSensorContentFileName, factorScoresFileName, numberOfUsers=6):
    for sensorID, sensor in sensors.items():
        Path(pictOutputFolder + contentID + '/' + sensor[0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
        filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID, userIDColumnName)
        print(filteredUserList)
        lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, factorScoresFileName, selectedFactor, numberOfUsers)
        
        for sensorFileNameExt, signalList in sensor[1].items():
            for signalName in signalList:
                generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(rootFolder, sensors, contentID, selectedFactor, usersDictFileName, outputFolderName,
                                                                          lowFactorUserIDs,
                                                            highFactorUserIDs,
                                                            sensorID,
                                                            signalName,
                                                            sensorFileNameExt) 