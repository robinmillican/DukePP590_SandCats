from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta
from scipy.stats import ttest_ind

main_dir = "/Users/Peizhi/Desktop/Duke 2015 SPRING/PUBPOL 590/data/"
root = main_dir + "3_task_data/"

##creating vectors
df = pd.read_csv(root + "allocation_subsamp.csv")
indx = df['tariff'] == 'E'
df[indx]
indxa1 = ((df['tariff'] == 'A') & (df['stimulus'] == '1'))
df[indxa1]
indxa3 = ((df['tariff'] == 'A') & (df['stimulus'] == '3'))
df[indxa3]
indxb1 = ((df['tariff'] == 'B') & (df['stimulus'] == '1'))
df[indxb1]
indxb3 = ((df['tariff'] == 'B') & (df['stimulus'] == '3'))
df[indxb3]

##set random seed to 1789 & use random choice to extract sample size)
np.random.seed(1789)
sampEE = np.random.choice(df[indx]['ID'], 300, replace = False)
sampA1 = np.random.choice(df[indxa1]['ID'], 150, replace = False)
sampA3 = np.random.choice(df[indxa3]['ID'], 150, replace = False)
sampB1 = np.random.choice(df[indxb1]['ID'], 50, replace = False)
sampB3 = np.random.choice(df[indxb3]['ID'], 50, replace = False)

## creat dataframe with all sample IDs
ids = sampEE.tolist() + sampA1.tolist() + sampA3.tolist() + sampB3.tolist() + sampB1.tolist()
df_ids = DataFrame(ids, columns = ['ID'])

##import and merge consumption data
df_consumption = pd.read_csv(root + "kwh_redux_pretrial.csv")
df_sampconsump = pd.merge(df_ids, df_consumption)

##group and aggregate by monthly
grp = df_sampconsump.groupby(['ID','month' ])
df_agg = grp['kwh'].sum()
df_agg1 = df_agg.reset_index()

##pivot by month
df_agg1['kwh_month'] = 'kwh_' + df_agg1.month.apply(str)
df_agg1_piv = df_agg1.pivot('ID', 'kwh_month', 'kwh')
df_agg1_piv.reset_index(inplace=True)
df_agg1_piv.columns.name = None
df_agg1_piv

##import and merge allocation data
df_allocation = pd.read_csv(root + "allocation_subsamp.csv")
df_sampallocate = pd.merge(df_allocation,df_agg1_piv)

##dummies
df_sampallocate['dummy'] = df_sampallocate.tariff.apply(str) + df_sampallocate.stimulus.apply(str)
df1 = pd.get_dummies(df_sampallocate, columns = ['dummy'])
kwh_cols = [v for v in df1.columns.values if v.startswith('kwh')]

##logit for a1
y = df1[(df1['dummy_A1']==1)|(df1['dummy_EE']==1)]
y = y['dummy_A1']

X = df1[(df1['dummy_A1']==1)|(df1['dummy_EE']==1)]
X = X[kwh_cols]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
logit_results = logit_model.fit() 
print(logit_results.summary())
 
##logit for a3
y = df1[(df1['dummy_A3']==1)|(df1['dummy_EE']==1)]
y = y['dummy_A3']

X = df1[(df1['dummy_A3']==1)|(df1['dummy_EE']==1)]
X = X[kwh_cols]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
logit_results = logit_model.fit() 
print(logit_results.summary()) 

##logit for b1
y = df1[(df1['dummy_B1']==1)|(df1['dummy_EE']==1)]
y = y['dummy_B1']

X = df1[(df1['dummy_B1']==1)|(df1['dummy_EE']==1)]
X = X[kwh_cols]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
logit_results = logit_model.fit() 
print(logit_results.summary()) 

##logit for b3
y = df1[(df1['dummy_B3']==1)|(df1['dummy_EE']==1)]
y = y['dummy_B3']

X = df1[(df1['dummy_B3']==1)|(df1['dummy_EE']==1)]
X = X[kwh_cols]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
logit_results = logit_model.fit() 
print(logit_results.summary()) 