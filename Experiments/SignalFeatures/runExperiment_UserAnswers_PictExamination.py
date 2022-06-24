# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/"

userAnswers_File = "Paper2 Data Eng Questions-Signals/UserAnswers/Experiment2_BoxesConvertedToAdIDColumns.csv"

df = pd.read_csv(rootFolder+userAnswers_File)




sns.catplot(x = "AdID",       # x variable name
            y = "aQ_75",       # y variable name
            hue = "AdID",  # group variable name
            data = df,     # dataframe to plot
            kind = "bar")