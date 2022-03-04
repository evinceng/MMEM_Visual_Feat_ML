# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:55:06 2021

@author: evinao
"""

import glob, os
import os.path
from pathlib import Path
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')
import pandas as pd
from shutil import copyfile


contents = ['C1','C2','C3','C4'] #, 'C2','C3','C4'

sensors = ['empatica', 'shimmer', 'tobii']

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/SignalCorrelation_29-12-2021/"

outputFile = "C:/Users/evinao/Documents/Paper2Data/UserFeaturesR3_v4.csv"

uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]

columnNameList = ['uID', 'AdID', 'AE', 'RE', 'AA', 'PI']

subscaleFolder = 'AE'

allFeatures_df = pd.DataFrame(data=uIDlist, columns=['uID'])

# features_dfs_list = [allFeatures_df]

strList = []
name = ''

result = pd.DataFrame()


# def getAllUsersScores(userIDList, factorScoresFileName, selectedFactor):
#     scores_df = pd.read_csv(factorScoresFileName)
#     scores_df = scores_df.loc[scores_df['userID'].isin(userIDList)] 
#     return scores_df[['userID', selectedFactor]]

def copyFeatureFiles(rootFolder, content, sensor, allFeatures_df, subscaleFolder):
    os.chdir(rootFolder + subscaleFolder + '/' + content + '/' +sensor)
    allFeatures_df['AdID'] = content[1]
    for file in glob.glob("*.csv"):
        print(file)
        if "FeatureVals_" in file and 'all_users_cutSignal' in file:
            df = pd.read_csv(file)
            
            strList = file.split()
            name = strList[1] + ' ' + strList[-2] #content + ' ' + sensor + ' ' 
            
            cols = df.columns
            for col in cols:
                if col not in columnNameList:
                    print(col)
                    df.rename({col:name}, axis=1, inplace =True)            
            # features_dfs_list.append(df)
            allFeatures_df = pd.merge(allFeatures_df, df[['uID', name]], on='uID', how='left')
            #assign content id number to adID 
    return allFeatures_df    
def copyFromAllContentFolders(rootFolder, allFeatures_df, subscaleFolder):
    contentFeatures_df = pd.DataFrame(data=[])
    for content  in contents:        
        print(content)
        for sensor in sensors:
            print(sensor)
            allFeatures_df = copyFeatureFiles(rootFolder, content, sensor, allFeatures_df, subscaleFolder)
        #  concatanete each content/id features adding new rows.
        contentFeatures_df = pd.concat([contentFeatures_df, allFeatures_df])
        allFeatures_df = pd.DataFrame(data=uIDlist, columns=['uID'])
        
    # result = pd.concat(features_dfs_list, axis=1, join="outer").drop_duplicates().reset_index(drop=True)
    contentFeatures_df.to_csv(outputFile, index=False)

copyFromAllContentFolders(rootFolder, allFeatures_df, subscaleFolder)