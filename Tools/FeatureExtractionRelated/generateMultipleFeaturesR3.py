# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:21:04 2021

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
contentList = [ 'C3', 'C4', 'C1','C2',] #  
isOnlyLowHighScoreUsersList = [True] #False, 

isOnlyLowHighScoreUsersQ = True # True
isOneUserQ = False #True
rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/"

outputFolderBaseName = "SignalCorrelation_29-12-2021/"
# outputFolder = "SignalCorrelation/"

factorScoresFileName = rootFolder + "MetaData/MMEM_Scores/mmem_C1_argmin_F_df.csv"

userFolder = rootFolder + "PsyUserSignals/user"
usersDictFileName = rootFolder + "MetaData/usersDict.xlsx"

userSensorContentFileName =  rootFolder + "MetaData/userContentSensorDict.csv"

selectedFactor = 'AE' # It is 'AE'
# contentID = 'C1'



numberofLowHighUsers = 12

sensorID = 1

higherSamplingFreqAndFolderExtension = 128

# Settings
lowpassQ = True
scatterQ = True
pValsQ = True
pValuesFileName = outputFolderBaseName + '/AllPValues.csv'  #+ contentID 

plotRawQ = False
plotScatteQ = True
plotNormParsQ = True
plot3D = False

writeMeanStdToFilesQ = True

# Configuration

preproc_meth = 'lowpass'

feature_codes = ['std', 'kurtosis', 'slope', 'num_of_peaks', 'total_var', 'spec_phs', 'spec_pow', 'num_of_peaks', 'slope', 'total_var']
#'mean', 


doTransformationQ = True #True
doNormalizationQ = True
cutSignalsQ = True
savecutDFQ = False
cutdfFolder = 'CutDF/'

predef_pupildiameter_mean = 3.0


feature_pars = []
feature_function = ''


saveStdMeanKurtosisQ = False

# not copied part
for selectedFactor in factorList:
    outputFolder = rootFolder + outputFolderBaseName + selectedFactor + "/"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    pValuesFileName = outputFolder + '/AllPValues.csv'  
    
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
   # end of not copied part                     
                    # Load MMAES scores (Copy here)
     
                    lowFactorUserIDs, highFactorUserIDs =  [], []

                    if isOnlyLowHighScoreUsersQ:
                        
                        lowFactorUserIDs, highFactorUserIDs = fct.getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, contentID, sensors, sensorID, feature_code, userSensorContentFileName, factorScoresFileName, numberofLowHighUsers)  
                        low_high_Fname= outputFolder + contentID + '/' + sensors[sensorID][0] + "/" + sensors[sensorID][0] + '_'
                        lowFactorUserIDs.to_csv(low_high_Fname+"lowFactorUserIDs.csv", index= False)
                        highFactorUserIDs.to_csv(low_high_Fname+"highFactorUserIDs.csv", index= False)
                        print(lowFactorUserIDs)
                        print(highFactorUserIDs)
                        
                        # Build dictionary
                        users = {}
                        for cuID in lowFactorUserIDs['uID']:
                            users[cuID] = {}   
                            users[cuID][sensorID] = {}
                            users[cuID][sensorID][signalName] = {}
                            users[cuID][selectedFactor] = lowFactorUserIDs.loc[lowFactorUserIDs['uID']==cuID][selectedFactor].iloc[0]
                            
                        for cuID in highFactorUserIDs['uID']:
                            users[cuID] = {}   
                            users[cuID][sensorID] = {}
                            users[cuID][sensorID][signalName] = {}
                            users[cuID][selectedFactor] = highFactorUserIDs.loc[highFactorUserIDs['uID']==cuID][selectedFactor].iloc[0]
                    
                    
                    else:
                        allusers = fct.getAllFilteredUserScores(outputFolder, selectedFactor, contentID, sensors, sensorID, feature_code, userSensorContentFileName, factorScoresFileName)
                                                                
                    
                        users = {}
                        for cuID in allusers['uID']:
                            users[cuID] = {}   
                            users[cuID][sensorID] = {}
                            users[cuID][sensorID][signalName] = {}
                            users[cuID][selectedFactor] = allusers.loc[allusers['uID']==cuID][selectedFactor].iloc[0]
                # end Load MMAES scores    
                    # Load data-------------------------------------------------------------------
                    # uIDs = lowFactorUserIDs['uID'].tolist() + highFactorUserIDs['uID'].tolist()
                    
                    # import Tools.signal_analysis_tools as sat
                    
                    
                    
                    lowFactorUsersMeanStd = []
                    highFactorUsersMeanStd = []
                    allUsersMeanStd = []
                    
                    if isOnlyLowHighScoreUsersQ:
                        uIDs = lowFactorUserIDs['uID'].tolist() + highFactorUserIDs['uID'].tolist()
                    else:
                        uIDs = allusers['uID']
                    
                    #store normalization parameters ['mean']
                    
                    
                    # Loop by users
                    for cuID in uIDs:
                        
                        # Get signal data 
                        signal_x_df = fct.getSignalDf(userFolder, higherSamplingFreqAndFolderExtension, cuID, sensors, sensorID, signalName, list(sensors[sensorID][1].keys())[0])
                        sig_x = np.array(signal_x_df[signalName])
                        sig_t = np.array(signal_x_df['timestamp_s'])
                        
                        
                        # Preprocessing
                        if lowpassQ:
                            sig_lp_x = sat.lowpass_1D(sig_x, cut_f)
                        else:
                            sig_lp_x = sig_x
                        
                        
                        # normalization: take the mean of signals while in room, and substract mean from each user/signal 
                        if doNormalizationQ:
                            
                            #covert pixel to mm for pupillabs
                            # if sensors[cuID] == 'pupillabs':
                            #     sig_lp_x = sig_lp_x * (3.5/32) # 3.5 non pupil labs mean, 32 is pupillabs mean
                            
                            sig_lp_x = sat.normalize_signal_by_background(users, usersDictFileName, cuID, sensorID, signalName, sig_t, sig_x, sig_lp_x, lowFactorUserIDs, selectedFactor, lowFactorUsersMeanStd, highFactorUsersMeanStd, allUsersMeanStd, predef_pupildiameter_mean, isOnlyLowHighScoreUsersQ)
                            
                            
                        time_int = []
                        if doTransformationQ:
                            #get min and max from the all signal
                            #print(cuID)
                            min_tr, max_tr = sat.get_scaling_pars(sig_t, sig_lp_x, time_int, feature_pars, code='standard', scaling_par=[0.03])
                            #print('min is ' + str(min_tr))
                            #print('max is ' + str(max_tr))
                            sig_p_x = sat.do_linear_transform(sig_t, sig_lp_x, time_int, min_tr, max_tr)
                        else:
                            sig_p_x = sig_lp_x
                        
                        #todo: Evin add ad video lines
                        if plotRawQ:
                            
                            sat.plot_raw_signals(outputFolder, contentID, sig_t, sig_x, sig_p_x, cuID, sensors, sensorID, signalName)
                            
                        
                        if cutSignalsQ:
                            time_int = sat.getAdTimeInterval(contentUserAdStartEndTimesFile, cuID)
                            if savecutDFQ:
                                fct.saveNumpyArraysToDf(sig_t, sig_x, sig_p_x, time_int, contentID, cuID, sensors, sensorID, signalName, cutdfFolder)
                            
                        # Feature extraction
                        # Set new feature extraction here [Incl 01]
                        
                        
                        
                        if feature_code == 'Gram_AF' or feature_code == 'Recurr_M' or feature_code == 'Markov_TF'or feature_code == 'Dyn_Time_W':
                            print('2d feature')
                            users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_2Dfeature(sig_t, sig_p_x, time_int, feature_pars, feature_code)
                        
                        if (feature_code == 'spec_phs' or feature_code == 'spec_pow') and feature_function == 'getF1oF23':   
                            users[cuID][sensorID][signalName][feature_code+'_F1oF23'] = sat.getF1oF23(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
                            users[cuID][sensorID][signalName][feature_code+'_F2oF13'] = sat.getF2oF13(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
                            users[cuID][sensorID][signalName][feature_code+'_F3oF12'] = sat.getF3oF12(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))    
                            
                        elif feature_function == 'getF1oF23':
                            print('getF1oF23') 
                            users[cuID][sensorID][signalName][feature_code] = sat.getF1oF23(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
                        elif feature_function == 'getF2oF3':
                            print('getF2oF3')
                            users[cuID][sensorID][signalName][feature_code] = sat.getF2oF3(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
                        elif feature_function == 'getF1oF2':
                           print('getF1oF2')
                           users[cuID][sensorID][signalName][feature_code] = sat.getF1oF2(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
                    
                        elif feature_function == 'feature1':
                           print('feature1')
                           users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)[0]
                    
                        elif feature_function == 'len':
                           print('len')
                           res = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)
                           users[cuID][sensorID][signalName][feature_code] = [len(res)]
                        
                        elif feature_function == 'getFxoFy':
                           print('getFxoFy')
                           x = [2]
                           y = [1]
                           users[cuID][sensorID][signalName][feature_code] = sat.getFxoFy(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code), x , y)
                        else:
                            # if cuID !=34:
                            users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)
                                # if "ECG_" in feature_code and len(users[cuID][sensorID][signalName][feature_code]) !=0:
                                #     np.savetxt('ECG_R_peaks_R2/uID_' + str(cuID) + '_HR_C1_' +feature_code +'.csv', np.asarray(users[cuID][sensorID][signalName][feature_code]))
                        if isOneUserQ:
                            break
                    if (feature_code == 'spec_phs' or feature_code == 'spec_pow')  and feature_function == 'getF1oF23':    
                        fct.writeFeatureToDf(users, outputFolder, contentID, sensors, sensorID, signalName, feature_code+'_F1oF23', selectedFactor, isOnlyLowHighScoreUsersQ, cutSignalsQ)
                        fct.writeFeatureToDf(users, outputFolder, contentID, sensors, sensorID, signalName, feature_code+'_F2oF13', selectedFactor, isOnlyLowHighScoreUsersQ, cutSignalsQ)
                        fct.writeFeatureToDf(users, outputFolder, contentID, sensors, sensorID, signalName, feature_code+'_F3oF12', selectedFactor, isOnlyLowHighScoreUsersQ, cutSignalsQ)
                    else:  
                        fct.writeFeatureToDf(users, outputFolder, contentID, sensors, sensorID, signalName, feature_code, selectedFactor, isOnlyLowHighScoreUsersQ, cutSignalsQ)
                    # df  = readFeatureDf(outputFolder, contentID, sensorID, signalName, feature_code, isOnlyLowHighScoreUsersQ, cutSignalsQ)
                    
                      
                    #write mean and std to files
                    low_mean_df, high_mean_df, mean_df = 0, 0, 0
                    if writeMeanStdToFilesQ:
                        meanColNames = ['uID','inside_room_Mean', 'inside_room_Std', 'inside_room_Kurtosis',selectedFactor]
                        meanFileNameBase = outputFolder + contentID + '/' + sensors[sensorID][0] +'/Mean_Std_Kurtosis_inside_room_' + sensors[sensorID][0] + '_'+  list(sensors[sensorID][1].keys())[0] + '_' + signalName + '_' + selectedFactor 
                        if isOnlyLowHighScoreUsersQ:
                            low_mean_df = pd.DataFrame(data=lowFactorUsersMeanStd, columns= meanColNames)
                            low_mean_df.to_csv(meanFileNameBase + '_low.csv', index=False)
                        
                            high_mean_df = pd.DataFrame(data=highFactorUsersMeanStd, columns=meanColNames)
                            high_mean_df.to_csv(meanFileNameBase +'_high.csv', index=False)
                        else:
                            mean_df = pd.DataFrame(data=allUsersMeanStd, columns=meanColNames)
                            mean_df.to_csv(meanFileNameBase +'_all.csv', index=False)
                            
                            
                    #plot a grapgh horizontal:uID, vertical: normalization parameter (mean) values(a dot grpah with points: colored by the exposure) 
                    if plotNormParsQ:
                        
                        sat.plot_norm_pars(outputFolder, contentID, sensors, sensorID, mean_df, low_mean_df, high_mean_df, selectedFactor, signalName, isOnlyLowHighScoreUsersQ)
                        
                        
                                       
                        
                    
                    if isOnlyLowHighScoreUsersQ and feature_code == 'spec_comp':
                        svt.generateSubPlotsofOneSignalOfMultipleUsers(users, sensors, contentID, selectedFactor, outputFolder,
                                                                        lowFactorUserIDs['uID'],
                                                                        highFactorUserIDs['uID'],
                                                                        sensorID,
                                                                        signalName,
                                                                        feature_code)
                        
                        
                    print(users)
                    
                   
                    
                    # P value and Visual inspection -------------------------------------------------------------------
                    coeff_types = ['Pearson', 'KendalTau']
                    # coeff_type = coeff_types[0]
                    
                    if (feature_code == 'spec_phs' or feature_code == 'spec_pow') and feature_function == 'getF1oF23':
                        feature_types = [feature_code+'_F1oF23', feature_code+'_F2oF13', feature_code+'_F3oF12']
                    
                    else:
                        feature_types = [feature_code]
                    
                    for feat in feature_types:
                        for coeff_type in coeff_types:
                        
                            fName = outputFolder + contentID + '/' + sensors[sensorID][0] + "/" + feat + '/' + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code + ' ' + coeff_type  
                            Path(outputFolder + contentID + '/' + sensors[sensorID][0] + "/" + feat).mkdir(parents=True, exist_ok=True)
                            # fName = outputFolder + contentID + '/' + sensors[sensorID][0] + "/" + feat + ' ' + coeff_type + ' '  + sensors[sensorID][0] + ' ' + signalName
                                
                            if isOnlyLowHighScoreUsersQ:
                                fName = fName + ' low-high_users'
                            else:
                                fName = fName + ' all_users'
                            
                            if cutSignalsQ:
                                fName = fName + '_cutSignal'
                            else:
                                fName = fName + '_allSignal'
                                
                            if pValsQ:
                                # if feat == 'slope' or feat == 'total_var' or feat == 'num_of_peaks' or feat == 'mean' or feat == 'std' or feat == 'kurtosis' or ('spec_phs' in feat): #or feature_code == 'spec_amp'   
                                x_data, y_data = [v[sensorID][signalName][feat][0] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()]  # userID [k for k,v in users.items()], 
                                # else:
                                #     x_data, y_data = [v[sensorID][signalName][feat] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()] # userID [k for k,v in users.items()], 
                                
                                r, p, es = sat.correlate_sigs_MME_OnlyData(x_data, y_data, coeff_type) #np.array(y_data, dtype='float'
                                
                                #generic effect size works with 2 classes
                                if isOnlyLowHighScoreUsersQ:  
                                    low_factor_userID_list = lowFactorUserIDs['uID'].tolist()
                                    if feat == 'slope' or feat == 'total_var' or feat == 'num_of_peaks' or feat == 'mean' or feat == 'std' or feat == 'kurtosis':
                                        gen_x_data, gen_y_data = [v[sensorID][signalName][feat][0] for k,v in users.items()], [0 if k in low_factor_userID_list else 1  for k,v in users.items()]
                                    else:
                                        gen_x_data, gen_y_data = [v[sensorID][signalName][feat] for k,v in users.items()], [0 if k in low_factor_userID_list else 1  for k,v in users.items()]
                                    
                                    genericEffectSize = sat.get_generic_es(np.array(gen_x_data), np.array(gen_y_data))
                                else: 
                                    genericEffectSize = 0
                                    
                                    
                                # Write tables
                                sat.write_tables_tex_csv(fName, pValuesFileName, feat, sensors, sensorID, signalName, cutSignalsQ, contentID, selectedFactor, coeff_type, r, p, es, genericEffectSize, isOneUserQ, isOnlyLowHighScoreUsersQ)
                                  
                                
                        # plot the mean and std dev of features, vertical lines
                        if plotScatteQ: # 2D scatter plot
                            sat.plot_scatter_sensor_factor(fName, users, uIDs, feat, sensors, sensorID, signalName, selectedFactor, isOneUserQ)
    
                        plt.close('all')
        