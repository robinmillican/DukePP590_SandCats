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

##slicing first 1million
df_subset = df[0:1000001]

## import SME and residential allocation values and rename columns
df_assign = pd.read_csv(root + "SME and Residential allocations.csv",usecols = ['ID','Code','Residential - Tariff allocation','Residential - stimulus allocation','SME allocation'],na_values = missing)
df1 = df_assign.rename(columns = {'ID':'panid'})
df1 = df1.rename(columns = {'Residential - Tariff allocation':'res_tariff'})
df1 = df1.rename(columns = {'Residential - stimulus allocation' :'res_stimulus'})

## Trimming data
df_assign = df[df_assign.Code == 1]
grp1 = df1.groupby(['Code','res_tariff','res_stimulus']) [['1','E','E']]
grp1.keys()
grp1['1'] 
grp1.values()[0] 
grp1.viewvalues()
[v for v in grp1.itervalues()]
grp1.values() # equivalent to above

[k for k in grp1.iterkeys()]
grp1.keys() # equivalent

[(k,v) for k,v in grp1.iteritems()]

grp1 = df1.groupby(['Code'])

##merge with with first 1 million
df2 = pd.merge(df_subset, df1, on = 'panid' )

## cleaning and dropping duplicates
df2.duplicated(['panid','date'])
df2.drop_duplicates(['panid','date'])

## splitting out hour column from date
df2['hh'] = df2['date']%100
df2 = df2[['panid','date','hh','kwh','Code','res_tariff','res_stimulus','SME allocation']]