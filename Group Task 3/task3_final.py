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
import statsmodels.api as sm

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
df_agg1['month_str'] = ['0' + str(v) if v < 10 else str(v) for v in df_agg1['month']]
df_agg1['kwh_month'] = 'kwh_' + df_agg1.month_str.apply(str)
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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# DEFINE FUNCTIONS -----------------
def ques_recode(srvy):

    DF = srvy.copy()
    import re
    q = re.compile('Question ([0-9]+):.*')
    cols = [unicode(v, errors ='ignore') for v in DF.columns.values]
    mtch = []
    for v in cols:
        mtch.extend(q.findall(v))

    df_qs = Series(mtch, name = 'q').reset_index() # get the index as a variable. basically a column index
    n = df_qs.groupby(['q'])['q'].count() # find counts of variable types
    n = n.reset_index(name = 'n') # reset the index, name counts 'n'
    df_qs = pd.merge(df_qs, n) # merge the counts to df_qs
    df_qs['index'] = df_qs['index'] + 1 # shift index forward 1 to line up with DF columns (we ommited 'ID')
    df_qs['subq'] = df_qs.groupby(['q'])['q'].cumcount() + 1
    df_qs['subq'] = df_qs['subq'].apply(str)
    df_qs.ix[df_qs.n == 1, ['subq']] = '' # make empty string
    df_qs['Ques'] = df_qs['q']
    df_qs.ix[df_qs.n != 1, ['Ques']] = df_qs['Ques'] + '.' + df_qs['subq']

    DF.columns = ['ID'] + df_qs.Ques.values.tolist()

    return df_qs, DF

def ques_list(srvy):

    df_qs, DF = ques_recode(srvy)
    Qs = DataFrame(zip(DF.columns, srvy.columns), columns = [ "recoded", "desc"])[1:]
    return Qs

# df = dataframe of survey, sel = list of question numbers you want to extract free of DVT
def dvt(srvy, sel):

    """Function to select questions then remove extra dummy column (avoids dummy variable trap DVT)"""

    df_qs, DF = ques_recode(srvy)

    sel = [str(v) for v in sel]
    nms = DF.columns

    # extract selected columns
    indx = []
    for v in sel:
         l = df_qs.ix[df_qs['Ques'] == v, ['index']].values.tolist()
         if(len(l) == 0):
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n\nERROR: Question %s not found. Please check CER documentation"
            " and choose a different question.\n" + bcolors.ENDC) % v
         indx =  indx + [i for sublist in l for i in sublist]

    # Exclude NAs Rows
    DF = DF.dropna(axis=0, how='any', subset=[nms[indx]])

    # get IDs
    dum = DF[['ID']]
    # get dummy matrix
    for i in indx:
        # drop the first dummy to avoid dvt
        temp = pd.get_dummies(DF[nms[i]], columns = [i], prefix = 'D_' + nms[i]).iloc[:, 1:]
        dum = pd.concat([dum, temp], axis = 1)
        # print dum

        # test for multicollineary

    return dum

def rm_perf_sep(y, X):

    dep = y.copy()
    indep = X.copy()
    yx = pd.concat([dep, indep], axis = 1)
    grp = yx.groupby(dep)

    nm_y = dep.name
    nm_dum = np.array([v for v in indep.columns if v.startswith('D_')])

    DFs = [yx.ix[v,:] for k, v in grp.groups.iteritems()]
    perf_sep0 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))
    perf_sep1 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(~DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))

    check = np.vstack([perf_sep0, perf_sep1])==0.
    indx = np.where(check)[1] if np.any(check) else np.array([])

    if indx.size > 0:
        keep = np.all(np.array([indep.columns.values != i for i in nm_dum[indx]]), axis=0)
        nms = [i.encode('utf-8') for i in nm_dum[indx]]
        print (bcolors.FAIL + bcolors.UNDERLINE +
        "\nPerfect Separation produced by %s. Removed.\n" + bcolors.ENDC) % nms

        # return matrix with perfect predictor colums removed and obs where true
        indep1 = indep[np.all(indep[nm_dum[indx]]!=1, axis=1)].ix[:, keep]
        dep1 = dep[np.all(indep[nm_dum[indx]]!=1, axis=1)]
        return dep1, indep1
    else:
        return dep, indep


def rm_vif(X):

    import statsmodels.stats.outliers_influence as smso
    loop=True
    indep = X.copy()
    # print indep.shape
    while loop:
        vifs = np.array([smso.variance_inflation_factor(indep.values, i) for i in xrange(indep.shape[1])])
        max_vif = vifs[1:].max()
        # print max_vif, vifs.mean()
        if max_vif > 30 and vifs.mean() > 10:
            where_vif = vifs[1:].argmax() + 1
            keep = np.arange(indep.shape[1]) != where_vif
            nms = indep.columns.values[where_vif].encode('utf-8') # only ever length 1, so convert unicode
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n%s removed due to multicollinearity.\n" + bcolors.ENDC) % nms
            indep = indep.ix[:, keep]
        else:
            loop=False
    # print indep.shape

    return indep


def do_logit(df, tar, stim, D = None):

    DF = df.copy()
    if D is not None:
        DF = pd.merge(DF, D, on = 'ID')
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        dum_cols = [v for v in D.columns.values if v.startswith('D_')]
        cols = kwh_cols + dum_cols
    else:
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        cols = kwh_cols

    # DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
    # set up y and X
    indx = (DF.tariff == 'E') | ((DF.tariff == tar) & (DF.stimulus == stim))
    df1 = DF.ix[indx, :].copy() # `:` denotes ALL columns; use copy to create a NEW frame
    df1['T'] = 0 + (df1['tariff'] != 'E') # stays zero unless NOT of part of control
    # print df1

    y = df1['T']
    X = df1[cols] # extend list of kwh names
    X = sm.add_constant(X)

    msg = ("\n\n\n\n\n-----------------------------------------------------------------\n"
    "LOGIT where Treatment is Tariff = %s, Stimulus = %s"
    "\n-----------------------------------------------------------------\n") % (tar, stim)
    print msg

    print (bcolors.FAIL +
        "\n\n-----------------------------------------------------------------" + bcolors.ENDC)

    y, X = rm_perf_sep(y, X) # remove perfect predictors
    X = rm_vif(X) # remove multicollinear vars

    print (bcolors.FAIL +
        "-----------------------------------------------------------------\n\n\n" + bcolors.ENDC)

    ## RUN LOGIT
    logit_model = sm.Logit(y, X) # linearly prob model
    logit_results = logit_model.fit(maxiter=10000, method='newton') # get the fitted values
    print logit_results.summary() # print pretty results (no results given lack of obs)


#####################################################################
#                           SECTION 2                               #
#####################################################################

main_dir = "/Users/Peizhi/Desktop/Duke 2015 SPRING/PUBPOL 590/data/"
root = main_dir + "3_task_data/"

nas = ['', ' ', 'NA'] # set NA values so that we dont end up with numbers and text
srvy = pd.read_csv(root + 'Smart meters Residential pre-trial survey data.csv', na_values = nas)
df = pd.read_csv(root + 'data_section2.csv')

# list of questions
qs = ques_list(srvy)

# get dummies
sel = [200,310,450]
dummies = dvt(srvy, sel)

# run logit, optional dummies
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort() # make sure the order correct with .sort()
stimuli.sort()

for i in tariffs:
    for j in stimuli:
        do_logit(df, i, j, D = dummies)




