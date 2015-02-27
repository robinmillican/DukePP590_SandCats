from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Pathing----------------------
main_dir = "/Users/robin/Desktop/Big Data/Data/CER Electricity/"
root = main_dir + "raw data/"
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("File")]

# Importing and stacking--------------------
df = pd.concat([pd.read_table(v, sep = " ",
    names = ['ID', 'DAYHH', 'kwh']) for v in paths], ignore_index = True)
df_allocation = pd.read_csv(root + "SME and Residential allocations.csv",
    usecols = ['ID','Code','Residential - Tariff allocation','Residential - stimulus allocation'],
    na_values = ['-', 'NA', 'NULL', '', '.'])
df2 = df_allocation.rename(columns = {'Residential - Tariff allocation':
    'RES_Tariff','Residential - stimulus allocation':'RES_Stimulus'})

# Trimming data before merging--------------------
df2 = df2[df2.Code <=1] ## Keeping only residential homes under "Code"
df2[(df2['RES_Tariff'] == 'A') & (df2['RES_Stimulus']== '1') | (df2['RES_Stimulus']== 'E')] 
df2 = df2[(df2['RES_Tariff'] == 'A') & (df2['RES_Stimulus']== '1') | 
    (df2['RES_Stimulus']== 'E')] ## Keeping only Tariff A and bi-monthly stimulus (1) or control (E)

# Merging on ID--------------------
df = pd.merge(df, df2, on = 'ID')

# Splitting the DAYHH column into Day and HH--------------------
df['hour_cer'] = df['DAYHH'] %100
df['day_cer'] = (df['DAYHH']-df['hour_cer']% 100)/100
df = df[['ID', 'DAYHH', 'hour_cer', 'day_cer', 'kwh', 'Code', 'RES_Tariff', 'RES_Stimulus']]

# Importing the time series correction--------------------
df_correction = pd.read_csv(root + "timeseries_correction.csv", 
    usecols = ['hour_cer', 'day_cer','date', 'year', 'month', 'day'])

# CER anomaly correction--------------------
df_correction.ix[df_correction['day_cer'] == 452, 'hour_cer']= np.array([v for 
    v in range(1,49) if v not in [2,3]])
df3 = pd.merge(df, df_correction, on= ['hour_cer','day_cer'])

# Aggregation by day --------------------
grp = df3.groupby(['ID','day_cer', 'RES_Stimulus']) #aggregating kwh consumption for every HH
agg = grp['kwh'].sum()
grp.sum()  
agg = agg.reset_index() # Resetting the index 
grp1 = agg.groupby(['day_cer','RES_Stimulus']) #reducing the amount of groups 
    #and increasing the list size
    #use agg.head() to look at first five rows

## Splitting up treatment/control--------------------
trt = {(k[0]): agg.kwh[v].values for k, v in grp1.groups.iteritems() 
    if k[1] == '1'} # set of all treatments by date
ctrl = {(k[0]): agg.kwh[v].values for k, v in grp1.groups.iteritems()
    if k[1] == 'E'} # set of all controls by date
keys = ctrl.keys()

# Calculating t-stats and p-values--------------------
tstats = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[0])))
    for k in keys], columns=['day_cer', 'tstat'])
pvals = DataFrame([(k, (ttest_ind(trt[k], ctrl[k], equal_var=False)[1]))
    for k in keys], columns=['day_cer', 'pval'])
t_p = pd.merge(tstats, pvals)

#Plotting daily t-stats and p-values--------------------
fig1 = plt.figure() # initializing plot
ax1 = fig1.add_subplot(2,1,1) # two rows, one column, first plot
ax1.plot(t_p['day_cer'],t_p['tstat']) #t-stat plot
ax1.axhline(2, color='r', linestyle='--')
ax1.axvline(180, color='g', linestyle='--')
ax1.set_title('t-stats over-time (daily)')

ax2 = fig1.add_subplot(2,1,2) # two rows, one column, second plot
ax2.plot(t_p['day_cer'], t_p['pval'])# adding p-value plot to Figure 1
ax2.axhline(0.05, color='r', linestyle='--')
ax2.axvline(180, color='g', linestyle='--')
ax2.set_title('p-values over-time (daily)')
plt.show()

# Aggregation by month
grp2 = df3.groupby(['ID','year','month','RES_Stimulus'])
agg1 = grp2['kwh'].sum()
agg1 = agg1.reset_index() # drop the multi-index
grp3 = agg1.groupby(['year', 'month', 'RES_Stimulus'])  #reducing the amount of groups 
#agg.head() to look at first five rows

## Splitting up treatment/control
trt1 = {(k[0], k[1]): agg1.kwh[v].values for k, v in grp3.groups.iteritems()
    if k[2] == '1'} # get set of all treatments by year-month
ctrl1 = {(k[0], k[1]): agg1.kwh[v].values for k, v in grp3.groups.iteritems()
    if k[2] == 'E'} # get set of all controls by year-month
keys2 = ctrl1.keys()

# Calculating t-stats and p-values
tstats1 = DataFrame([(k, np.abs(float(ttest_ind(trt1[k], ctrl1[k],
    equal_var=False)[0]))) for k in keys2], columns=['month', 'tstat1'])
pvals1 = DataFrame([(k, (ttest_ind(trt1[k], ctrl1[k],
    equal_var=False)[1])) for k in keys2], columns=['month', 'pval1'])
t_p1 = pd.merge(tstats1, pvals1)

# Plotting monthly t-stats and p-values
fig2 = plt.figure() # initialize plot
ax3 = fig2.add_subplot(2,1,1) # two rows, one column, first plot
ax3.plot(t_p1['tstat1'])
ax3.axhline(2, color='r', linestyle='--')
ax3.axvline(6, color='g', linestyle='--')
ax3.set_title('t-stats over-time (monthly)') # plotting t-stats

ax4 = fig2.add_subplot(2,1,2) # two rows, one column, second plot
ax4.plot(t_p1['pval1'])
ax4.axhline(0.05, color='r', linestyle='--')
ax4.axvline(6, color='g', linestyle='--')
ax4.set_title('p-values over-time (monthly') # adding plot of p-values
plt.show()