"""9/9 pts"""

from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


##============Section 0
main_dir =  '/Users/dnoriega/Dropbox/pubpol590_sp15/data_sets/CER/tasks/4_task_data/'

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
##================ section 3

main_dir =  '/Users/dnoriega/Dropbox/pubpol590_sp15/data_sets/CER/tasks/4_task_data/'
df = pd.read_csv(main_dir + 'task_4_kwh_long.csv')

os. chdir(main_dir)
from fe_functions import *

df = pd.merge(df_w, df)
df['TP'] = df['trt'].apply(int) * df.trial.apply(int)

df['log_kwh'] = (df['kwh']+1).apply(np.log)
df['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df['month']])
df['ym'] = df['year'].apply(str) + "_" + df['mo_str']

y = df['log_kwh']
P = df['trial']
TP = df['TP']
w = df['w']
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1]
X = pd.concat([TP, P, mu], axis=1)
ids = df['ID']
y = demean(y, ids)
X = demean(X, ids)

##model without weights
fe_model = sm.OLS(y, X)
fe_results = fe_model.fit()
print(fe_results.summary())


##model with weights
y = y*w
nms = X.columns.values
X = np.array([x*w for k, x in X.iteritems()])
X = X.T
X = DataFrame(X, columns = nms)

fe_w_model = sm.OLS(y, X)
fe_w_results = fe_w_model.fit()
print(fe_w_results.summary())






