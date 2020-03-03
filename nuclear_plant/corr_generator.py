#%%
import numpy as np
import pandas as pd
import os
import glob
from functools import partial
import multiprocessing
from multiprocessing import Pool
from module.data_loader_v2 import data_loader

import warnings 
import random
import joblib
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
#%%
train_path = 'data/train/'
test_path = 'data/test/'
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
label = pd.read_csv('data/train_label.csv')
# #%%
# def data_loader_all(func, files, folder='', train_label=None):
#     func_fixed = partial(func, folder=folder, train_label=train_label)
#     if __name__ == '__main__':
#         pool = Pool(processes=multiprocessing.cpu_count()) 
#         df_list = list(pool.imap(func_fixed, files)) 
#         pool.close()
#         pool.join()        
#     combined_df = pd.concat(df_list)    
#     return combined_df

# # train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label)
# test = data_loader_all(data_loader, test_list, folder=test_path, train_label=None)

#%% 30 train set
# b = train.loc[30,train.dtypes =='object'].T
# non = train.loc[train.index!=30]
# b[30] = non.loc[:,non.dtypes=='object'].mean().values
# b= b.T
# train.loc[30,train.dtypes =='object'] = b
# train.iloc[:,1:-1] = train.iloc[:,1:-1].astype('float64')
#%% 1154/1168 testset
# for i in [1154,1168]:

#     b = test.loc[i,test.dtypes =='object'].T
#     non = test.loc[(test.index!=1154)&(test.index!=1168)]
#     b[i] = non.loc[:,non.dtypes=='object'].mean().values
#     b= b.T
#     test.loc[i,test.dtypes =='object'] = b
# test.iloc[:,1:-1] = test.iloc[:,1:-1].astype('float64')
#%%


#%%
# train.to_pickle('preprocess/train')
# test.to_pickle('preprocess/test')
#%%
train =pd.read_pickle('preprocess/train')
# test = pd.read_pickle('preprocess/test')

# #%%
# train = train.loc[:,train.nunique()>30]
# #%%
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(train.iloc[:,1:-1])
# b= train.copy()
# b.iloc[:,1:-1] = scaled
# #%%
# col = b.iloc[:,1:-1].std().sort_values(ascending=False).tail(200).index
# col = pd.DataFrame(col)
# col.to_csv('preprocess/inv_col_list.csv', index=False)
#%%

# 1661-4000 rjsrkdqhgjaskqqnghkrdlstj 6rodnjf
# 055 741 2933
#%%
col = pd.read_csv('preprocess/inv_col_list.csv')
col = col.iloc[:,0].values
#%%
# col = np.arange(len(train.loc[0].columns)-2)
mask = np.ones((len(col),len(col)),dtype='bool')
mask[np.triu_indices(len(col))] = False

def corr_generator(flist):
    data = train.loc[flist][col]
    if flist in [1154, 1168]:
        for i in range(len(data)-5):
            tcorr = np.array(data.iloc[i:i+5,:].corr())[mask]
            if i == 0:
                table = tcorr
            else :
                table = np.vstack([table,tcorr])

    else :    
        for i in range(len(data)-15):
            tcorr = np.array(data.iloc[i:i+15,:].corr())[mask]
            if i == 0:
                table = tcorr
            else :
                table = np.vstack([table,tcorr])
    
    # dlabel = label.loc[flist]['label'] 
    # table = np.hstack([table,np.repeat(dlabel, len(table)).reshape(-1,1)])
    # table = pd.DataFrame(table)
    # table = table[col_corr]
    # table = table.dropna(axis=1)
    # table = table.loc[:,(table.std()>0.01)&(table.std()<0.05)]
  
    return table


#%%
def corr_merger(flist):   
    func_fixed = partial(corr_generator)     
    
    if __name__ == '__main__':
        pool = Pool(processes=32) 
        corr_list = list(pool.imap(func_fixed, flist)) 
        pool.close()
        pool.join()        
    combined_corr = np.vstack(corr_list)    
    return combined_corr
#%%
inv_train_corr = corr_merger(np.arange(828))
inv_train_corr = pd.DataFrame(inv_train_corr)
inv_train_corr.to_pickle('preprocess/inv_train_corr')
# %%
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
# X = train_corr
# Y = np.repeat(label['label'].values,30)
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

# model = xgb.XGBClassifier(n_jobs=-1,objective='multi:softprob', n_estimators=100,
#     max_depth=3, learning_rate=0.1)
# eval_set=[(X_train, Y_train), (X_test, Y_test)]
# model.fit(X_train,Y_train , eval_metric=["merror","mlogloss"], eval_set=eval_set, verbose=True)
# #%%

# #%%
# dtrain = xgb.DMatrix(X, label=Y)
# params = {'objective':'multi:softprob','colsample_bytree':0.3,'learning_rate':0.1,
#                 'max_depth':5, 'alpha':10,'num_class':198,'n_jobs':-1}

# cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=5,num_boost_round=50, early_stopping_rounds=10, metrics='mlogloss')  

# # dtrain = xgb.DMatrix(X_train, label=Y_train)
# # dvalid = xgb.DMatrix(X_test, label=Y_test)
# # dtest = xgb.DMatrix(test[feature_names].values)
# # watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
# #%%
# train_corr

# %%
