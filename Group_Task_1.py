from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
##_______________________________________________
## import files, stack and merging
main_dir = "/Users/Peizhi/Desktop/Duke 2015 SPRING/PUBPOL 590/"
root = main_dir + "raw/"
paths = [ root + "File" + str(v) + ".txt" for v in range(1,7) ]
missing = ['.', 'NA', 'NULL', '','999999999','9999999']
list_of_dfs = [ pd.read_table(v,sep = " ",names = ['panid', 'date', 'kwh'],na_values = missing) for v in paths]
len(list_of_dfs)
type(list_of_dfs)
type(list_of_dfs[0])

df = pd.concat(list_of_dfs, ignore_index = True)
missing = ['.', 'NA', 'NULL', '','999999999','9999999']

## import SME and residential allocation values and rename columns
df_assign = pd.read_csv(root + "SME and Residential allocations.csv",usecols = ['ID','Code','Residential - Tariff allocation','Residential - stimulus allocation','SME allocation'],na_values = missing)
df1 = df_assign.rename(columns = {'ID':'panid'})
df1 = df1.rename(columns = {'Residential - Tariff allocation':'res_tariff'})
df1 = df1.rename(columns = {'Residential - stimulus allocation' :'res_stimulus'})
df2 = pd.merge(df, df1, on = 'panid' )

## cleaning and dropping duplicates
df2.duplicated(['panid','date'])
df2.drop_duplicates(['panid','date'])

## splitting out hour column from date
df2['hh'] = df2['date']%100
df2 = df2[['panid','date','hh','kwh','Code','res_tariff','res_stimulus','SME allocation']]


## prolem1: DST missing/ extra entries
## problem2: floor size "999999999"

## Potential problem in the dataset or 
## 1. The decision of whether keeping/dropping daylight saving values need to wait until the actual analysis. 
## 2. group 3 may need to be dropped in further analysis since they did not complete the trial
## 3. adding dummy variables for different groups may be more convinient in next-step analysis
## 4. If they are group 1(residential), do they  have missing data for any other answers in the survey?

