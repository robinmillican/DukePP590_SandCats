from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

 
##============Section 0 
main_dir =  "/Users/Peizhi/Desktop/Duke 2015 SPRING/PUBPOL 590/data/task4/"

os. chdir(main_dir)
from logit_functions import *

df = pd.read_csv(main_dir + '14_B3_EE_w_dummies.csv')
df = df.dropna(axis=0, how='any')

tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()


drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)

##!! df_logit how is it different with df_pretrial
for i in tariffs:
    for j in stimuli:
        # dummy vars must start with "D_" and consumption vars with "kwh_"
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)


##================== by hand:

df_mean = df_logit.groupby('tariff').mean().transpose()


df_s = df_logit.groupby('tariff').std().transpose()
df_n = df_logit.groupby('tariff').count().transpose().mean()
top = df_mean['B'] - df_mean['E']
bottom = np.sqrt(df_s['B']**2/df_n['B'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'



##================Section 1
df = pd.read_csv(main_dir + 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna(axis=0, how='any')

tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()


drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)


for i in tariffs:
    for j in stimuli:
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)
  
grp = df_logit.groupby('tariff')
df_mean = grp.mean().transpose()

df_mean = df_logit.groupby('tariff').mean().transpose()
df_mean.C - df_mean.E

df_s = df_logit.groupby('tariff').std().transpose()
df_n = df_logit.groupby('tariff').count().transpose().mean()
top = df_mean['C'] - df_mean['E']
bottom = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'      

        
##===============Section 2

df_logit['p_hat'] = logit_results.predict()
df_logit['trt'] = 0 + (df_logit['tariff'] == 'C')


df_logit['w'] = np.sqrt((df_logit['trt']/df_logit['p_hat'] + (1-df_logit['trt'])/ (1- df_logit['p_hat']) ))

df_w = df_logit[['ID','trt','w']]