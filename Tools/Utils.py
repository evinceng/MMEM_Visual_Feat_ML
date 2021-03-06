# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:37:21 2021

@author: TTTT
"""

import json
from datetime import datetime, timedelta
import pandas as pd
import dateutil
from tzlocal import get_localzone
import os.path, time
import os,glob
from pathlib import Path
import pickle
import numpy as np
# on MM

mm_seq = {}
mm_seq['Box1'] = ['C4', 'C3', 'C2', 'C1']
mm_seq['Box4'] = ['C1', 'C2', 'C3', 'C4']
mm_seq['Box2'] = ['C3', 'C1', 'C4', 'C2']
mm_seq['Box3'] = ['C2', 'C4', 'C1', 'C3']

# C1-L_Explore- H_Littlebaby
# C2-M2_72kg- H_DietCoke
# C3-M1_Tahiti- M_Bounty
# C4-H_Mixtape-L_Waring

video_len = {} 
video_len['C1'] = '00:04:04' #C1-L_Explore- H_Littlebaby
video_len['C2'] = '00:03:48' #C2-M2_72kg- H_DietCoke
video_len['C3'] = '00:03:54' #C3-M1_Tahiti- M_Bounty
video_len['C4'] = '00:03:38' #C4-H_Mixtape-L_Waring

ad_times = {}
ad_times['C1'] = ['00:00:50:060', '00:01:56:080']
ad_times['C2'] = ['00:01:57:230', '00:02:29:280']
ad_times['C3'] = ['00:02:08:010', '00:02:43:120']
ad_times['C4'] = ['00:00:43:090', '00:01:58:170']


raw_data_root_folder = 'D:/LivingLabMeasurements/'

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/"
usersDictFileName = rootFolder + "MetaData/usersDict.xlsx"

filesFoldersDictFileName = rootFolder + "MetaData/livinglabUsersFileFolderNames.xlsx"

outputFolder = "GeneratedMetaData/"

absStartTimeOfSensorsFileName = 'usersAbsStartTimes_Tobii_Added.xlsx'

adStartEndTimeFileName = "_usersAdStartEndTimes.csv"

def loadFigFromPickleFile(filename):
    figx = pickle.load(open(filename, 'rb'))
    figx.show() # Show the figure, edit it, etc.!
    
def writePickleFile(fig, filename):
    pickle.dump(fig, open(filename +'.pickle', 'wb'))


# @brief Convert time string to a seconds from middnight (miliseconds are decimals of seconds)
# @arg time_str time string in supported forms
# @arg date0_str optional, ignored 
def get_secs_from_str(time_str, date0_str='20210514'):
    
    if len(time_str) <= 1:
        return 0
    
    if time_str[0] == '-':
        sign_mul = -1.0
        time_str = time_str[1:]
    else:
        sign_mul = 1.0
        

    if (':' in time_str) and not ('-' in time_str):
        num_of_cc = time_str.count(':')
        if num_of_cc == 2:
            dt = datetime.strptime(date0_str + ' ' + time_str, '%Y%m%d %H:%M:%S').time()
            return sign_mul*(60*60*dt.hour + 60*dt.minute + dt.second)
        elif num_of_cc == 3:
            dt = datetime.strptime(date0_str + ' ' + time_str, '%Y%m%d %H:%M:%S:%f').time()
            return sign_mul*(60*60*dt.hour + 60*dt.minute + dt.second + dt.microsecond/1000000.0)
    elif '-' in time_str:
        if '.' in time_str:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f').time()
            return sign_mul*(60*60*dt.hour + 60*dt.minute + dt.second + dt.microsecond/1000000.0)
        else:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').time()
            return sign_mul*(60*60*dt.hour + 60*dt.minute + dt.second)
    else:
        dt = datetime.strptime(time_str, '%Y%m%d%H%M%S%f').time()
        return sign_mul*(60*60*dt.hour + 60*dt.minute + dt.second + dt.microsecond/1000000.0)


#t_str = '2021-05-25 09:17:42.709'
#t_s = get_secs_from_str(t_str)
#print (t_s)


video_len_sec = {k: get_secs_from_str(v) for k, v in video_len.items()}
ad_times_sec = {k: [get_secs_from_str(v[0]), get_secs_from_str(v[1])] for k, v in ad_times.items()}

# @brief compute video and ad times for a given user
# @par uID user uID
# @return a list of times [start_video, start_ad, end_ad, end_video]
def get_video_and_ad_times_fromJsonFile(usersJsonFileName, uID):   
    
    #dictionary keys are only allowed to be string, this is why uID is converted to string
    uID_str = str(uID)
    
    # Opening JSON file
    with open(usersJsonFileName) as json_file:
        users_dict = json.load(json_file)
  
    # print(users_dict)    
    uID_Box = users_dict[uID_str]['Box']
    
    print(uID_Box)
    uID_mm_seq = mm_seq[uID_Box]
    
    out_times_lst = []
    for ii in range(4):
        C_ii = uID_mm_seq[ii]
        c_start_video_time = get_secs_from_str(users_dict[uID_str]['R' + str(ii+1)])
        c_end_video_time = c_start_video_time + video_len_sec[C_ii] # Add times properly!
        c_start_ad_time = get_secs_from_str(users_dict[uID_str]['R' + str(ii+1)]) + ad_times_sec[C_ii][0]
        c_end_ad_time = get_secs_from_str(users_dict[uID_str]['R' + str(ii+1)]) + ad_times_sec[C_ii][1]
    
        out_times_lst.append([c_start_video_time, c_start_ad_time, c_end_ad_time, c_end_video_time])
        print(out_times_lst)
    return out_times_lst


# @brief compute video and ad times for a given user
# @par uID user uID
# @return a list of times [start_video, start_ad, end_ad, end_video]
def get_video_and_ad_times(usersDictFileName, uID):   
    
    users_dict = readDict(usersDictFileName)
  
    # print(users_dict)    
    uID_Box = users_dict[uID]['Box']
    
    print(uID_Box)
    uID_mm_seq = mm_seq['Box' + str(uID_Box)]
    
    out_times_lst = []
    for ii in range(4):
        C_ii = uID_mm_seq[ii]
        c_start_video_time = get_secs_from_str(users_dict[uID]['R' + str(ii+1)])
        c_end_video_time = c_start_video_time + video_len_sec[C_ii] # Add times properly!
        c_start_ad_time = get_secs_from_str(users_dict[uID]['R' + str(ii+1)]) + ad_times_sec[C_ii][0]
        c_end_ad_time = get_secs_from_str(users_dict[uID]['R' + str(ii+1)]) + ad_times_sec[C_ii][1]
    
        out_times_lst.append([c_start_video_time, c_start_ad_time, c_end_ad_time, c_end_video_time])
        # print(out_times_lst)
    return out_times_lst


# @brief compute video and ad times for a given user
# @par 
# @return a list of times [start_video, start_ad, end_ad, end_video]
def get_video_and_ad_times_userslist_one_content(usersDictFileName, uIDList, contentID='C1'):   
    
    users_dict = readDict(usersDictFileName)   
    # print(users_dict)
    
    out_times_lst = []
    
    for uID in uIDList:
        uID_Box = 'Box' + str(users_dict[uID]['Box'])# Box1
        print(str(uID) + ' ' + uID_Box)
        C_ii = mm_seq[uID_Box].index(contentID)
        roundID = 'R' + str(C_ii + 1)
        c_start_video_time = get_secs_from_str(users_dict[uID][roundID])
        c_end_video_time = c_start_video_time + video_len_sec[contentID] # Add times properly!
        c_start_ad_time = c_start_video_time + ad_times_sec[contentID][0]
        c_end_ad_time = c_start_video_time + ad_times_sec[contentID][1]
    
        out_times_lst.append([c_start_video_time, c_start_ad_time, c_end_ad_time, c_end_video_time])
        print(out_times_lst)
    return out_times_lst

# @brief compute video and ad times for a given user
# @par 
# @return a list of times [start_video, start_ad, end_ad, end_video]
def get_video_and_ad_times_userslist_one_content_uid_boxID(usersDictFileName, uIDList, contentID='C1'):   
    
    users_dict = readDict(usersDictFileName)   
    # print(users_dict)
    
    out_times_lst = []
    
    for uID in uIDList:
        uID_Box = 'Box' + str(users_dict[uID]['Box'])# Box1
        print(str(uID) + ' ' + uID_Box)
        C_ii = mm_seq[uID_Box].index(contentID)
        roundID = 'R' + str(C_ii + 1)
        c_start_video_time = get_secs_from_str(users_dict[uID][roundID])
        c_end_video_time = c_start_video_time + video_len_sec[contentID] # Add times properly!
        c_start_ad_time = c_start_video_time + ad_times_sec[contentID][0]
        c_end_ad_time = c_start_video_time + ad_times_sec[contentID][1]
    
        out_times_lst.append([uID,users_dict[uID]['Box'], users_dict[uID][roundID], c_start_video_time, c_start_ad_time, c_end_ad_time, c_end_video_time])
        print(out_times_lst)
    return out_times_lst

#test if works
# uID = 1
# usersJsonFileName = 'users_1_9_36.json'
# out_times_lst = get_video_and_ad_times(usersJsonFileName, uID)
# print(out_times_lst)

def getFloatFromOneOrTwoDotsTimestamp(val):
    return float(".".join(val.split(".", 2)[:2]))

def utcToLocalTimeZone(utc_time, formatStr = "%Y-%m-%d %H:%M:%S"):
    date_time_obj = datetime.utcfromtimestamp(utc_time)
    # localize to UTC first
    date_time_obj = date_time_obj.replace(tzinfo=dateutil.tz.UTC)
    
    # now localize to timezone of the localzone:
    new_timestamp = date_time_obj.astimezone(get_localzone())
    
    new_timestamp_str = datetime.strftime(new_timestamp, formatStr)
    
    return new_timestamp_str


def readDict(dictFilePath):
    df = pd.read_excel(dictFilePath)
    df.set_index('uID',  inplace=True)
    dict = df.to_dict(orient='index')
    
    return dict

#get start time form filename
def getShimmerStartTimeFromFileName(fileName):
    if pd.isna(fileName):
        return '00:00:00'
    # from pathlib import Path
    # date_time_str = Path(shimmerFileName).name.split(" ")[0]
    date_time_str = fileName.split(" ")[0]
    date_time_str = date_time_str.split('/')[-1]
    date_time_obj = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
    # date_time_str = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    date_time_str = date_time_obj.strftime("%H:%M:%S")
    print(date_time_str)
    return date_time_str

def getEmpaticaStartTimeOfFromFolderName(empaticaFolderpath):
    #The first row is the initial time of the session expressed as unix timestamp in UTC.
    #get timestamp as float
    timestamp_begin_utc = empaticaFolderpath.split("_")[0]
    timestamp_begin_utc = timestamp_begin_utc.split("/")[-1]
    startTime = utcToLocalTimeZone(int(timestamp_begin_utc), formatStr="%H:%M:%S")
    print(startTime)
    return startTime

def getPupilLabsStartTimeFromFolderName(fileName):
    date_time_str = fileName.split("/")[-4]
    date_time_obj = datetime.strptime(date_time_str, '%Y%m%d%H%M%S%f')
    # date_time_str = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    date_time_str = date_time_obj.strftime("%H:%M:%S")
    print(date_time_str)
    
    return date_time_str


def getHikVisionCreationTime(hikvisionFilePath):
    creationTime = time.ctime(os.path.getctime(hikvisionFilePath))
    date_time_str = creationTime.split(' ')[-2]
    print(date_time_str)
    return date_time_str
    
# # go to calibrations/latest folder/calibration.json file modified date +2
# def getTobiiCreationTimeFromCalibrationJsonFile(tobiiFilePath, subFolder):
#     tobii_Folder = os.path.join(tobiiFilePath, subFolder)
#     calibrationFilesList = []
#     for root, dirs, files in os.walk(tobii_Folder):
#         for name in files:
#             if name == 'calibration.json':
#                 calibrationFilesList.append(name)
    
#     created_Time = max(glob.glob(os.path.join(tobii_Folder, '*/')), key=os.path.getmtime)
#     # created_Time = created_Time + timedelta(hours=2)
#     print(created_Time)
#     return created_Time
    
def getTobiiStartTimeFromFilesMaxTime(tobiiFilePath):
    fnList = ["accelerometer.csv", "gazeDirection.csv", "gazePosition.csv",
              "gazePosition3D.csv", "gyroscope.csv", "pupilCenter.csv", "pupilDim.csv"]
    startTimes = []
    for fileName in fnList:
        if os.path.isfile(tobiiFilePath + fileName):
            tobii_df = pd.read_csv(tobiiFilePath + fileName)
            startTime_sec = get_secs_from_str(tobii_df["utc_time"].iloc[0])
            startTimes.append(startTime_sec)
        else:
            print(tobiiFilePath + fileName + " doesn't exist")
    
    if startTimes:
        return max(startTimes)
    else:
        print("There is not tobii files")
        return 0
    
def fillAbsStartTimesOfSensors(raw_data_root_folder, filesFoldersDictFileName):
   
    data_fn_dict = readDict(filesFoldersDictFileName)
    users = {}
    
    for uID in data_fn_dict.keys():
        users[uID] = {}
        users[uID]['Box'] = data_fn_dict[uID]['Box']
        users[uID]['empaticaAbsStartTime'] = getEmpaticaStartTimeOfFromFolderName(data_fn_dict[uID]['empaticaFolderPath'])
        users[uID]['empaticaAbsStartTime_sec'] = get_secs_from_str(users[uID]['empaticaAbsStartTime'])
        users[uID]['shimmerAbsStartTime'] = getShimmerStartTimeFromFileName(data_fn_dict[uID]['shimmerFilePath'])
        users[uID]['shimmerAbsStartTime_sec'] = get_secs_from_str(users[uID]['shimmerAbsStartTime'])
        
        #the hikvision creation time is not ok with only user36, the first user. otherwise correct.
        if(uID == 36):
            users[uID]['hikVisionAbsStartTime'] = '11:58:09'
        else:
            users[uID]['hikVisionAbsStartTime'] = getHikVisionCreationTime(raw_data_root_folder + data_fn_dict[uID]['hikVisionFilePath'])
        users[uID]['hikVisionAbsStartTime_sec'] = get_secs_from_str(users[uID]['hikVisionAbsStartTime'])
        
        print(uID)
        
        if(not data_fn_dict[uID]['isTobii']):
            users[uID]['pupillabsAbsStartTime'] = getPupilLabsStartTimeFromFolderName(data_fn_dict[uID]['pupilLabsFolderPath'])
            users[uID]['pupillabsAbsStartTime_sec'] = get_secs_from_str(users[uID]['pupillabsAbsStartTime'])
        else:
            if uID == 50:
                continue;
            if(not pd.isnull(data_fn_dict[uID]['tobiiFolderPath_csharp'])):
                users[uID]['tobiiAbsStartTime_sec'] = getTobiiStartTimeFromFilesMaxTime(raw_data_root_folder + data_fn_dict[uID]['tobiiFolderPath_csharp'])
            else: 
                print("no charp data")
              
        
        
    df = pd.DataFrame.from_dict(users, orient='index') # convert dict to dataframe
    df.index.name = 'uID'
    
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    df.to_excel(outputFolder + absStartTimeOfSensorsFileName) # write dataframe to file

    print(users)




def writeAdStartAndEndTimesToFile(usersDictFileName):
    uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]
    
    columns = ["uID", "BoxID", "RoundStart", "VS", "AS", "AE", "VE"]
               # ,"C2_VS", "C2_AS", "C2_VE", "C2_AE",
               # "C3_VS", "C3_AS", "C3_VE", "C3_AE","C4_VS", "C4_AS", "C4_VE", "C4_AE"]
    contents = ["C1", "C2", "C3", "C4"]
    data = []
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    for content in contents:
        out_times_lst = get_video_and_ad_times_userslist_one_content_uid_boxID(usersDictFileName, uIDlist, contentID=content)
        # numpy_array = np.array(out_times_lst)
        # transpose = numpy_array.T        
        # transpose_list = transpose.tolist()
        # data=uIDlist + out_times_lst
        df = pd.DataFrame(data=out_times_lst, columns=columns)
        df.to_csv(outputFolder + content + adStartEndTimeFileName, index=False)
        
def getAdTimeInterval(fileName, uID):
    times_df = pd.read_csv(fileName)
    
    return [times_df[uID]['AS'],times_df[uID]['AE']]

# # test fillAbsStartTimesOfSensors
# fillAbsStartTimesOfSensors(raw_data_root_folder, filesFoldersDictFileName)

# #create the when contents started and ended for each user
# writeAdStartAndEndTimesToFile(usersDictFileName)
# # end of create the when contents started and ended for each user

# # test functions
# getTobiiStartTimeFrompupilDim(root_folder + "user18/Tobii/")
# getTobiiCreationTimeFromCalibrationJsonFile(root_folder + "user18/Tobii", "calibrations")
# getShimmerStartTimeFromFileName("user36/Shimmer/20210506114816 Device5E7F.csv")
# getEmpaticaStartTimeOfFromFolderName("user36/Empatica/1620294325_A0264A/")
# getPupilLabsStartTimeFromFolderName("user36/Pupillabs/20210506115642217/exports/001/")
# getHikVisionCreationTime("F:/LivingLabMeasurements/user1/user1_ceiling.mp4")

# #run for testing code
# filesFoldersDictFileName = "Data/livinglabUsersFileFolderNames.xlsx"
# data_fn_dict = readDict(filesFoldersDictFileName)
# print(data_fn_dict)
# print('\n')
# print(data_fn_dict[1])

# #run for testing code
# usersDictFileName = "Data/usersDict.xlsx"
# users = readDict(usersDictFileName)
# print(users)
# print('\n')
# print(users[1])


# print(get_secs_from_str("00:06:45:080"))
# print(get_secs_from_str("00:02:54:800"))
# print(get_secs_from_str("00:05:42:200"))

# usersDictFileName = "Data/usersDict.xlsx"
# get_video_and_ad_times(usersDictFileName, 14)

# loadFigFromPickleFile("C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user8/uID-8 shimmer empatica abs acc.pickle")

# loadFigFromPickleFile("C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user2/uID-2 shimmer empatica abs acc.pickle")

# loadFigFromPickleFile("C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user2/uID-2_subplots.pdf.pickle")