# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:35:23 2021

@author: evinao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_openml
import scipy.stats as scs
import pandas as pd
from sklearn.metrics import roc_curve, plot_roc_curve, auc, f1_score
from sklearn.metrics import RocCurveDisplay
#%% Functions

# @brief merge all classes to class 0 but the sel_C set to 1   
def merge_cls(y_labels, sel_C):
    
    
    t_tr_labels = pd.DataFrame(y_labels, columns=['L'])
    y_merged_labels = t_tr_labels['L'].transform(lambda x: 1 if x == sel_C else 0)
    
    return np.array(y_merged_labels)


# @brief assign clasees to cont. features:
def get_cls_from_feature1D(feats, cls_ints):
    
    feats_ar = np.array(feats)
    f_df = pd.DataFrame(feats_ar, columns=['L'])
    n = len(feats)
    M = len(cls_ints)+1
    
    y_labels = np.zeros(n) # First class is denoted by zero
    y_labels += f_df['L'].transform(lambda x: 1 if x < cls_ints[0] else 0)
    for cii in range(1, M-1):    
        y_labels += (cii+1)*f_df['L'].transform(lambda x: 1 if (cls_ints[cii-1] <= x and x < cls_ints[cii]) else 0)
    y_labels += M*f_df['L'].transform(lambda x: 1 if cls_ints[M-2] < x else 0) # Above last T
    
    return y_labels.astype(int)

#feats = 1 + 5*np.random.rand(10)
#cls_ints = [2,3]
#y_labels = get_cls_from_feature1D(feats, cls_ints)
#print (y_labels)

#%% 
# @brief plot ROC curve for multiclass 
def plot_ROCs(y_true, y_predict, pos_label=1):
    
    cls_lbs = np.unique(y_true)
    M = len(cls_lbs)
    
    plt.figure()
    lw = 2

    if M == 2: # Two class
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict, pos_label=pos_label)
        roc_AUC = auc(fpr, tpr)
        F1 = f1_score(y_true, y_predict, average='macro')
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f, F1 = %0.2f)' % (roc_AUC, F1))
        
    if M > 2: # More than two class
        
        for sel_C in cls_lbs:
            
           y_true_mrg = merge_cls(y_true, sel_C)
           y_predict_mrg = merge_cls(y_predict, sel_C)
           
           fpr, tpr, thresholds = metrics.roc_curve(y_true_mrg, y_predict_mrg, pos_label=1)
           roc_AUC = auc(fpr, tpr)
           F1 = f1_score(y_true_mrg, y_predict_mrg)
           plt.plot(fpr, tpr, lw=lw, label='Cls=%0.0f, AUC = %0.2f, F1 = %0.2f)' % (sel_C, roc_AUC, F1))
            
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
    return 0



# @brief gets generic effect size
# @arg x 1D array of data
# @arg y labels od groups, two classes only
def get_generic_es(x, y):
    
    labels = np.unique(y)
    M = len(labels)
    if M != 2:
        return 0
    
    x_G1 = x[y==labels[0]]
    x_G2 = x[y==labels[1]]
    
    n1, n2 = len(x_G1), len(x_G2)
    mu1, mu2 = np.mean(x_G1), np.mean(x_G2)
    sd1, sd2 = np.std(x_G1), np.std(x_G2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1*sd1 + (n2-1)*sd2*sd2) / (n1 + n2 - 2))

    gen_es = np.abs(mu2-mu1) / pooled_sd
    
    return gen_es

#x = np.array([1,2,3,4,5,6,5,4,3,2,1]) # data
#y = np.array([2,3,3,2,2,3,2,3,2,2,3]) # labels
#p_sd = get_generic_es(x, y)
#print (p_sd)


# @brief compute P, R, F class by class for a given conf mat
# @arg CM: confusion matrix
# 
def get_PRF_from_CM(CM):
    
    return 1


# @brief return expected confusion matrix of a random classifier
# @arg cls_sizes_lst a list of classification class sizes [n1, n_2, ...]
def get_rand_cls_CM(cls_sizes_lst):
    M = len(cls_sizes_lst)
    CM = np.zeros((M, M))
    for ii in range(M):
        CM[ii, :] = (cls_sizes_lst[ii]/M)*np.ones(M)
    return CM
#cls_sizes_lst=  [12, 10, 20]
#CM = get_rand_cls_CM(cls_sizes_lst)
#print (CM)

# @brief get p-values per class of a given confusion matrix with H0=[classification is random]
def get_pvals_percls_from_CM(CM):
    
    M = CM.shape[0]
    rand_CM = get_rand_cls_CM(CM.sum(axis=1))
    p_vals = np.zeros(M) 
    
    for ii in range(M):
        
        # Perform Chi-Square Goodness of Fit Test
        observed = CM[ii, :]
        expected = rand_CM[ii, :]
        s, p = scs.chisquare(f_obs=observed, f_exp=expected)
        p_vals[ii] = p
        
        # UseZ-test - might be stronger
        
    return p_vals
#CM = np.array([[20, 11, 6], [8, 19, 3], [11, 2, 21]])
#p_vals = get_pvals_percls_from_CM(CM)
#print (p_vals)





#%% Settings
fileName = 'Data/UserFeaturesR3_v2_1.csv'

columnNames = ['uID', 'AdID', 'BoxID', 'AE_LH', 'RE_LH', 'AA_LH', 'PI_LH', 'AE', 'RE', 'AA', 'PI',
               'AccX kurtosis', 'AccX mean', 'AccX num_of_peaks', 'AccX slope',
               'AccX spec_phs_F1oF23', 'AccX spec_phs_F2oF13', 'AccX spec_phs_F3oF12',
               'AccX spec_pow_F1oF23', 'AccX spec_pow_F2oF13', 'AccX spec_pow_F3oF12',
               'AccX std', 'AccX total_var', 'EDA kurtosis', 'EDA mean', 'EDA num_of_peaks',
               'EDA slope', 'EDA spec_phs_F1oF23', 'EDA spec_phs_F2oF13', 'EDA spec_phs_F3oF12',
               'EDA spec_pow_F1oF23', 'EDA spec_pow_F2oF13', 'EDA spec_pow_F3oF12', 'EDA std',
               'EDA total_var', 'HR kurtosis', 'HR mean', 'HR num_of_peaks', 'HR slope', 
               'HR spec_phs_F1oF23', 'HR spec_phs_F2oF13', 'HR spec_phs_F3oF12', 'HR spec_pow_F1oF23',
               'HR spec_pow_F2oF13', 'HR spec_pow_F3oF12', 'HR std', 'HR total_var',
               'SC kurtosis', 'SC mean', 'SC num_of_peaks', 'SC slope', 'SC spec_phs_F1oF23',
               'SC spec_phs_F2oF13', 'SC spec_phs_F3oF12', 'SC spec_pow_F1oF23', 'SC spec_pow_F2oF13', 
               'SC spec_pow_F3oF12', 'SC std', 'SC total_var', 'Temp kurtosis', 
               'Temp mean', 'Temp num_of_peaks', 'Temp slope', 'Temp spec_phs_F1oF23',
               'Temp spec_phs_F2oF13', 'Temp spec_phs_F3oF12', 'Temp spec_pow_F1oF23',
               'Temp spec_pow_F2oF13', 'Temp spec_pow_F3oF12', 'Temp std', 'Temp total_var',
               'diameter kurtosis', 'diameter mean', 'diameter num_of_peaks', 'diameter slope', 
               'diameter spec_phs_F1oF23', 'diameter spec_phs_F2oF13', 'diameter spec_phs_F3oF12',
               'diameter spec_pow_F1oF23', 'diameter spec_pow_F2oF13', 'diameter spec_pow_F3oF12',
               'diameter std', 'diameter total_var']

#AE_columns = ['HR std', 'Temp slope', 'EDA spec_phs_F1oF23', 'diameter mean', 'SC slope', 'HR slope'] #'HR spec_phs_F1oF23',  'diameter mean', 'SC slope', 'HR spec_phs_F3oF12',  'EDA spec_phs_F1oF23', 'EDA spec_phs_F3oF12'
AE_columns = ['HR std', 'Temp slope', 'diameter mean', 'SC slope', 'HR slope'] #'H
RE_columns = ['HR std', 'HR total_var', 'HR spec_pow_F2oF13','SC spec_phs_F3oF12', 'Temp slope', 'AccX spec_phs_F3oF12', 'EDA spec_pow_F2oF13',
              'AccX num_of_peaks',] #'HR spec_pow_F1oF23', 'HR spec_pow_F3oF12', 'EDA spec_pow_F3oF12', 

AA_columns = ['SC spec_phs_F1oF23', 'diameter slope', 'Temp slope', 'Temp total_var', 'HR spec_phs_F2oF13', 'AccX num_of_peaks', 'HR total_var',]

PI_columns = ['EDA spec_phs_F1oF23', 'HR spec_phs_F2oF13', 'AccX num_of_peaks', 'SC spec_phs_F3oF12', 'Temp mean',]

subscale_columns_dict = {'AE': AE_columns, 'RE': RE_columns, 'AA': AA_columns, 'PI': PI_columns}

subscl_label = 'AE'
# subscl_label = 'RE'
# subscl_label = 'AA'
# subscl_label = 'PI'
#%% Load data & set labels

# Load data
all_df = pd.read_csv(fileName)

# Select rows - users or videos
adID = 1
features_rows_df = all_df#[all_df['AdID']==adID]
group_rows_df = all_df[subscl_label]#[all_df['AdID']==adID]







# Select features
features_sel_df = features_rows_df[subscale_columns_dict[subscl_label]]
features_sel_df.fillna(features_sel_df.mean(), inplace=True)



# Standardize features
#features_st_df = (features_sel_df - features_sel_df.mean()) / features_sel_df.std()
features_st_df = features_sel_df #


# Selected
features_df = features_st_df
#plot histograms

# Get numpy array
features = np.array(features_df)


# Get specific cases selection
subSelction = 'Mid_out'
#subSelction = 'Mid_vs_rest'


if subSelction == 'Mid_out':
    n_T = 21
    group_sort_df = group_rows_df.sort_values()
    T1 = group_sort_df.iloc[n_T]
    T2 = group_sort_df.iloc[-n_T-1]
    
    select_sub = (group_rows_df <= T1) | (T2 <= group_rows_df)

    features_rows_df = features_df[select_sub]
    features = np.array(features_rows_df)
    
    
    group_rows_df = group_rows_df[select_sub]

    T_s = (T1 + T2) / 2.0
    cls_int = [T_s]
    
    group = get_cls_from_feature1D(group_rows_df, cls_int)
    

if subSelction == 'Mid_vs_rest':
    n_T = 21
    group_sort_df = group_rows_df.sort_values()
    T1 = group_sort_df.iloc[n_T]
    T2 = group_sort_df.iloc[-n_T-1]
    select_sub = (group_rows_df <= T1) | (T2 <= group_rows_df)
    not_select_sub = [not x for x in select_sub]
    
    group = np.zeros(len(group_rows_df))            
    group[select_sub] = 1
    group[not_select_sub] = 2
    
    
# Correlations
N = features.shape[1]
corrs_lst = np.zeros((2, N))
for ii in range(N):
    r, p = scs.kendalltau(features[:, ii], np.array(group_rows_df))
    corrs_lst[:, ii] = [np.round(r, 3), np.round(p, 3)]


'''
# Set labels
if subscl_label == 'AE':
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 0.625 else (2 if x < 1.25 else (3 if x < 1.875 else (4 if x < 2.5 else (5 if x < 3.125 else (6 if x < 3.75 else (7 if x < 4.375 else 8)))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.875 else (7 if x < 3.75 else 8))
       
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.77 else 2)
    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.5 else 2)
    #group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.65 else 2)
    
    
                      

    
    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.625 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.6 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.5 else (3 if x < 3.5 else (4 if x < 4.5 else 5))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.5 else (2 if x < 3.5 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3.5 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.5 else 2)    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3 else 2)  
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.75 else 2) 
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.56 else 2)    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.25 else 2)    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.1 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.04 else (2 if x < 2.56 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x <= 2.04 else (2 if x <= 2.56 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.52 else (2 if x < 2.04 else (3 if x < 2.56 else (4 if x < 3.08 else 5))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x <= 1.52 else (2 if x < 2.56 else 3 ))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x <= 2.04 else (2 if x < 3.08 else 3 ))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x <= 1.8 else (2 if x < 2.8 else 3 ))
    
    

elif subscl_label == 'RE':
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.5 else (3 if x < 3.5 else (4 if x < 4.5 else 5))))
    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.3 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.1 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.2 else 2)
    
    #group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.15 else 2)
    cls_int = [4.15]    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.16 else 2)
    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.125 else 2)    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 4.175 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x <= 1.88 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.5 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.0 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.44 else (2 if x < 1.88 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.88 else (2 if x < 2.32 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.6 else (2 if x < 2.0 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.0 else 3))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.5 else (3 if x < 3.5 else (4 if x < 4.5 else 5))))

    group = get_cls_from_feature1D(group_rows_df[subscl_label], cls_int)

elif subscl_label == 'AA':    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 0.625 else (2 if x < 1.25 else (3 if x < 1.875 else (4 if x < 2.5 else (5 if x < 3.125 else (6 if x < 3.75 else (7 if x < 4.375 else 8)))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 0.5 else (2 if x < 1.0 else (3 if x < 1.5 else (4 if x < 2.0 else (5 if x < 2.5 else (6 if x < 3.0 else (7 if x < 3.5 else (8 if x < 4.0 else (9 if x < 4.5 else 10)))))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3.0 else (2 if x < 3.1 else (3 if x < 3.2 else (4 if x < 3.3 else (5 if x < 3.4 else (6 if x < 3.5 else 7))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.5 else (2 if x < 2.6 else (3 if x < 2.7 else (4 if x < 2.8 else (5 if x < 2.9 else (6 if x < 3.0 else 7))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.1 else (2 if x < 2.2 else (3 if x < 2.3 else (4 if x < 2.4 else (5 if x < 2.5 else (6 if x < 3.0 else (7 if x < 3.1 else (8 if x < 3.2 else (9 if x < 3.3 else (10 if x < 3.4 else (11 if x < 3.5 else 7)))))))))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.5 else (3 if x < 3.5 else (4 if x < 4.5 else 5))))
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3.5 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.8 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.9 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.875 else 2)
    #group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.885 else 2)
    cls_int = [2.885]
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.8875 else 2)
    
    group = get_cls_from_feature1D(group_rows_df[subscl_label], cls_int)
    
elif subscl_label == 'PI':
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 0.625 else (2 if x < 1.25 else (3 if x < 1.875 else (4 if x < 2.5 else (5 if x < 3.125 else (6 if x < 3.75 else (7 if x < 4.375 else 8)))))))
    
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3.25 else 2)
    #group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3 else 2)
    cls_int = [3]
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 3.1 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.75 else 2)
    # group_df = all_df[subscl_label].transform(lambda x: 1 if x < 2.85 else 2)
    
    
    group = get_cls_from_feature1D(group_rows_df[subscl_label], cls_int)
    
# subscl_label = 'AE'
# group_df = all_df[subscl_label].transform(lambda x: 1 if x < 1.5 else (2 if x < 2.5 else (3 if x < 3.5 else (4 if x < 4.5 else 5))))

#group = np.array(group_df)

# X, y = datasets.load_iris(return_X_y=True)
'''

X, y = features, group


#clf = svm.SVC(kernel='linear', C=1, random_state=42)
#scores = cross_val_score(clf, X, y, cv=10)



#%% Confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeavePOut, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

selectedML = 'SVM_rbf'
#selectedML = 'SVM_poly' # kernel= {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
#selectedML = 'SVM_linear'
#selectedML = 'SVM_sigmoid'
#selectedML = 'Tree'
#selectedML = 'kNN'
#selectedML = 'LogReg'
#selectedML = 'NeuralNetMPL'
#selectedML = 'StochGrad'

# Simple version - not correct
# y_pred = cross_val_predict(clf, X, y, cv=10)
# conf_mat = confusion_matrix(y, y_pred)

y_pred_concat = []
y_test_concat = []
        
# Correct folding
M = len(np.unique(y))
all_CM = np.zeros((M,M))

print(np.unique(group))
# numberofSplits = len(np.unique(group))


# model_selection.GroupKFold([n_splits])
# model_selection.LeaveOneOut()
# model_selection.LeaveOneGroupOut()

#folds = KFold(n_splits=5, shuffle=True) #, random_state=0)
#folds = LeavePOut(5)
folds = ShuffleSplit(n_splits=5, test_size=.3) #, random_state=0)
for train_index, test_index in folds.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    # Get folds
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_concat = []

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    if selectedML == 'SVM_rbf':
        clf = svm.SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

    if selectedML == 'SVM_poly':
        clf = svm.SVC(kernel='poly')
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        
    if selectedML == 'SVM_linear':
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
       
    if selectedML == 'SVM_sigmoid':
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
    
        
    if selectedML == 'kNN':
        neigh = KNeighborsClassifier(n_neighbors=6)
        neigh.fit(X_train, y_train)
        y_predict = neigh.predict(X_test)
        
    
    if selectedML == 'Tree':
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)


    if selectedML == 'LogReg':
        tree = LogisticRegression()
        tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

    
    if selectedML == 'NeuralNetMPL':
        tree = MLPClassifier(max_iter=3000)
        tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)
        
        
    if selectedML == 'StochGrad':
        tree = SGDClassifier(max_iter=3000)
        tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)
        


    curr_CM = confusion_matrix(y_test, y_predict, labels=np.unique(group))
    
    y_pred_concat = [*y_pred_concat,*y_predict ]
    y_test_concat = [*y_test_concat, *y_test]
    
    print('===========')
    print(curr_CM)
    
    print('===========')
    # Sum confusion matrices
    if curr_CM.shape[0] == M:
        all_CM += curr_CM
    
print (all_CM)


#%% Get characteristics 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.decision_function(X_train)
R = recall_score(y_test, y_pred, average='micro')
P = precision_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')
print ('(P, R, F1) = ', (P, R, F1))



# PR curve
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
#y_score = clf.decision_function(X_train)
#prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
#pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


#%% ROC curve
# from sklearn.metrics import roc_curve, plot_roc_curve, auc, f1_score
# from sklearn.metrics import RocCurveDisplay






# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7) # y_concat
# clf = svm.SVC(random_state=0)
# clf.fit(X_train, y_train)

# y_true = y_test
# y_predict = clf.predict(X_test)

# # Get curve & stats Plot it
# plot_ROCs(y_true, y_predict)

plot_ROCs(y_test_concat, y_pred_concat)

# Print 
#print ('AUC: ', roc_AUC)
#print ('F1: ', F1)





   
   
#%% Significance of confusion matrix

# Goodnes of fit significant difference 
p_vals = get_pvals_percls_from_CM(all_CM)
print ('Class by class p_vals: ', p_vals)

