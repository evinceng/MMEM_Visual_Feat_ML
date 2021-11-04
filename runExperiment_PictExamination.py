# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:16:04 2021

@author: evinao
"""
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from datetime import datetime
import os.path
from pathlib import Path
import Tools.pictExamination_tools as pet

from signalDics import *


rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/"

factorScoresFileName = rootFolder +"MetaData/MMEM_Scores/mmem_C1_argmin_F_df.csv"
selectedFactor = 'AE'
contentID = 'C1'

userSensorContentFileName =  rootFolder +"MetaData/userContentSensorDict.csv"

usersDictFileName = rootFolder +"MetaData/usersDict.xlsx"

userFolder = rootFolder + "PsyUserSignals/user"

outputFolderName = "PictorialProof/"

userIDColumnName = 'uID'
# lowFactorUserIDs = [14,16,17,10] #tobii [14,16,17,10] # empatica, shimmer pupillabs [1,2,3,4]
# highFactorUserIDs = [50,51,52,53] # empattica, shimmer [5,6,7,8] # pupillabs [33,34,35,36] #tobii

sensorID = 1
sensorFileNameExt = 'ACC'# 'pupilCenter_left_eye'#'left_eye_2d' #EDA
signalName = 'AccX' #'pc_x' #'diameter' # 'EDA'


sensors = {1:['empatica', empaticaSignals] , 2:['shimmer', shimmerSignals],
           3:['tobii',tobiiSignals],  4:['pupillabs', pupillabsSignals]}
#sensors = {1:['empatica', empaticaSignals]}

# mmaesFactors = {1:'AE', 2:'RE', 3:'AA', 4:'PI'}
  

# create the figures for one sensor-signal 

# filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, 1, 1)
# print(filteredUserList)
# lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, factorScoresFileName, selectedFactor)
# generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(rootFolder, sensors, selectedContent, selectedFactor, usersDictFileName, outputFolderName,
#                                                             lowFactorUserIDs,
#                                                             highFactorUserIDs,
#                                                             sensorID,
#                                                             signalName,
#                                                             sensorFileNameExt)


pictOutputFolder = "PictorialProof/"
numberOfUsers = 6
pet.createFiguresForAll(userFolder, pictOutputFolder, contentID, selectedFactor, usersDictFileName, outputFolderName, userIDColumnName, sensors, userSensorContentFileName, factorScoresFileName)  

# to view already created pickle files:
    
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F2_empatica_HR_HR_2021_10_19-11-58-51.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_diameter_diameter_2021_10_19-12-59-13.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F2_empatica_ACC_AccX_2021_10_19-11-58-38.pickle')



# 6 users  
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccX_2021_09_07-12-32-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccY_2021_09_07-12-32-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccZ_2021_09_07-12-32-08.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_BVP_BVP_2021_09_07-12-32-14.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_EDA_EDA_2021_09_07-12-32-11.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_HR_HR_2021_09_07-12-32-17.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_X_m-(s^2)_2021_09_07-12-32-26.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_Y_m-(s^2)_2021_09_07-12-32-35.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_Z_m-(s^2)_2021_09_07-12-32-44.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_X_m-(s^2)_2021_09_07-12-34-35.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Y_m-(s^2)_2021_09_07-12-34-25.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Z_m-(s^2)_2021_09_07-12-34-16.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Z_m-(s^2)_2021_09_07-12-34-16.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Conductance_microSiemens_2021_09_07-12-32-53.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Conductance_uS-1_2021_09_07-12-33-03.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Resistance_kOhms_2021_09_07-12-33-39.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_X_deg-s_2021_09_07-12-33-58.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_Y_deg-s_2021_09_07-12-34-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_Z_deg-s_2021_09_07-12-33-48.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__PPG_A12_mV_2021_09_07-12-33-21.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Pressure_BMP280_kPa_2021_09_07-12-33-12.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Temperature_BMP280_Degrees Celsius_2021_09_07-12-33-31.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_normal0_x_2021_09_07-12-40-10.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_normal1_x_2021_09_07-12-40-34.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_point_3d_x_2021_09_07-12-39-46.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_norm_pos_x_2021_09_07-12-39-30.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_2d_diameter_2021_09_07-12-36-10.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_2d_ellipse_angle_2021_09_07-12-36-15.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_3d_circle_3d_normal_x_2021_09_07-12-37-15.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_3d_diameter_2021_09_07-12-36-52.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_x_2021_09_07-12-34-38.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_y_2021_09_07-12-34-41.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_z_2021_09_07-12-34-44.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazeDirection_left_eye_gd_x_2021_09_07-12-34-47.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition_gp_latency_2021_09_07-12-35-11.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition_gp_x_2021_09_07-12-35-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition3D_gp3d_x_2021_09_07-12-35-14.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gyroscope_gy_x_2021_09_07-12-35-24.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_pupilCenter_left_eye_pc_x_2021_09_07-12-35-33.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_pupilDim_left_eye_diameter_2021_09_07-12-35-53.pickle')


#4 users                                             
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_AccX_2021_09_06-00-29-59.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_BVP_2021_09_06-00-30-06.pickle')   
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_EDA_2021_09_06-00-30-05.pickle') 
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_HR_2021_09_06-00-30-08.pickle')    
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_HR_2021_09_06-00-30-08.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Conductance_microSiemens_2021_09_06-00-30-29.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Conductance_uS-1_2021_09_06-00-30-34.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Resistance_kOhms_2021_09_06-00-30-56.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Pressure_BMP280_kPa_2021_09_06-00-30-40.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_PPG_A12_mV_2021_09_06-00-30-45.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Temperature_BMP280_Degrees Celsius_2021_09_06-00-30-51.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Gyro_X_deg-s_2021_09_06-00-31-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Gyro_Z_deg-s_2021_09_06-00-31-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Accel_WR_Z_m-(s^2)_2021_09_06-00-31-18.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_normal0_x_2021_09_06-00-42-46.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_normal1_x_2021_09_06-00-43-00.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_point_3d_x_2021_09_06-00-42-33.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_norm_pos_x_2021_09_06-00-42-24.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_2d_diameter_2021_09_06-00-40-39.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_2d_ellipse_angle_2021_09_06-00-40-41.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_3d_circle_3d_normal_x_2021_09_06-00-41-13.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_3d_diameter_2021_09_06-00-41-00.pickle')


# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_x_2021_09_06-00-39-45.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_y_2021_09_06-00-39-47.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_z_2021_09_06-00-39-49.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazeDirection_left_eye_gd_x_2021_09_06-00-39-51.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition_gp_latency_2021_09_06-00-40-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition_gp_x_2021_09_06-00-40-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition3D_gp3d_x_2021_09_06-00-40-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gyroscope_gy_x_2021_09_06-00-40-13.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_pupilCenter_left_eye_pc_x_2021_09_06-00-40-18.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_pupilDim_left_eye_diameter_2021_09_06-00-40-29.pickle')








                                                  