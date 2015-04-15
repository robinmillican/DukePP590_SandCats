## section 3
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

 
main_dir =  "/Users/Peizhi/Desktop/Duke 2015 SPRING/PUBPOL 590/data/task4/"
df = pd.read_csv(main_dir + 'task_4_kwh_long.csv')

os. chdir(main_dir)
from fe_functions import *

df = pd.merge(df_w, df)
df['TP'] = df.trt.apply(str)+ df.trial.apply(str)

df['log_kwh'] = (df['kwh']+1).apply(log)
df['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df['month']])
df['ym'] = df['year'].apply(str) + "_" + df['mo_str']

y = df['log_kwh']
T = df['trt']
TP = df['TP']
w = df['w']
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1]
X = pd.concat([TP, T, mu], axis=1)
ids = df['ID']
y = demean(y, ids)
