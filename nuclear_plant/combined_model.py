#%%
print('****now process is working...****')
import numpy as np
import pandas as pd
import os
import glob
from functools import partial
import multiprocessing
from multiprocessing import Pool
from module.data_loader import data_loader
import math

import warnings 
import random
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA,KernelPCA



train_path = 'data/train/'
test_path = 'data/test/'
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
label = pd.read_csv('data/train_label.csv')

def data_loader_all(func, files, folder='', train_label=None, nrows=45, skiprows = range(1,16)):   
    func_fixed = partial(func, folder=folder, train_label=train_label, nrows=nrows, skiprows=skiprows)
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df

train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label, nrows=45, skiprows = range(1,16))
test = data_loader_all(data_loader, test_list, folder=test_path, train_label=None, nrows=45, skiprows=range(1,16))
print('data load completed')

train = train.loc[:,train.nunique()>2]
#%%
# def column_generator():
#     data = train.loc[0]
#     data = data.iloc[:,1:-1]
#     for i in range(30):
#         tcorr = np.array(data.iloc[i:i+15,col].corr())
#         if i == 0:
#             table = tcorr
#         else :
#             table = np.dstack([table,tcorr])
    
#     table = table.reshape(len(col)*len(col),30)
#     table = pd.DataFrame(table.astype('float16')).drop_duplicates()
#     ind = table.index
    
#     return list(ind)


# col = np.arange(len(train.columns)-2)
# random.seed(1206)
# col = random.sample(list(col),200)
# corr_index = column_generator()
# corr_inde = pd.DataFrame(corr_index)
# corr_inde.to_csv('preprocess/corr_index_v1.csv', index=False)

train_files = np.arange(828)
test_files = np.arange(828,1548)

# def corr_generator(flist):
#     data = train.loc[flist]
#     data = data.iloc[:,1:-1]
#     for i in range(30):
#         tcorr = np.array(data.iloc[i:i+15,col].corr())
#         if i == 0:
#             table = tcorr
#         else :
#             table = np.dstack([table,tcorr])
    
#     table = table.reshape(len(col)*len(col),30)
#     table = pd.DataFrame(table.astype('float16')).loc[corr_index].T
#     return table


# def corr_merger(flist):   
#     func_fixed = partial(corr_generator)     
    
#     if __name__ == '__main__':
#         pool = Pool(processes=multiprocessing.cpu_count()) 
#         corr_list = list(pool.imap(func_fixed, flist)) 
#         pool.close()
#         pool.join()        
#     combined_corr = pd.concat(corr_list)    
#     return combined_corr


# import gc
# gc.collect()

# train_corr = corr_merger(train_files)
# train_corr = train_corr.reset_index(drop=True)



# def corr_generator(flist):
#     data = test.loc[flist]
#     data = data.iloc[:,1:-1]
#     for i in range(30):
#         tcorr = np.array(data.iloc[i:i+15,col].corr())
#         if i == 0:
#             table = tcorr
#         else :
#             table = np.dstack([table,tcorr])
    
#     table = table.reshape(len(col)*len(col),30)
#     table = pd.DataFrame(table.astype('float16')).loc[corr_index].T
#     return table

# test_corr = corr_merger(test_files)
# test_corr = test_corr.reset_index(drop=True)
# print('corr_data is made completely')

# pca1 = PCA(n_components=500)
# pca_tf = pca1.fit_transform(pd.concat([train_corr.fillna(0),test_corr.fillna(0)]))
# combined_corr = pd.DataFrame(pca_tf)




label_list = []
for i in range(828):
    label_list.append(train['label'].loc[i].iloc[0])
label_list = np.repeat(label_list, 30)
#%%

rolled_train = train.iloc[:,1:-1].copy()
rolled_train = rolled_train.groupby(train.index).rolling(15).mean()
# rolled_train2 = rolled_train.groupby(train.index).rolling(15).std()
# rolled_train = pd.concat([rolled_train1,rolled_train2], axis=1)
# rolled_train.columns = [i for i in range(rolled_train.shape[1])]
rolled_train = rolled_train.dropna()
rolled_train = rolled_train.groupby(rolled_train.index).tail(30)
rolled_train = rolled_train.reset_index(drop=True)

rolled_test = test.iloc[:,1:].copy()
rolled_test = rolled_test.loc[:,rolled_train.columns]
rolled_test = rolled_test.groupby(rolled_test.index).rolling(15).mean()
rolled_test = rolled_test.dropna()
rolled_test = rolled_test.groupby(rolled_test.index).tail(30)
rolled_test = rolled_test.reset_index(drop=True)

#%%


# pca2 = PCA(n_components=2000)
# pca_tf2 = pca2.fit_transform(pd.concat([rolled_train, rolled_test]))
# combined = pd.DataFrame(pca_tf2)

# combined_data = pd.concat([combined_corr,combined], axis=1)
# combined_train = combined.iloc[:len(rolled_train),:] 
# combined_test  = combined.iloc[len(rolled_train):,:]

# combined_train = pd.concat([train_corr.fillna(0),rolled_train.iloc[:,1:-1]], axis=1)
# combined_test = pd.concat([test_corr.fillna(0),rolled_test], axis=1)
combined_train = rolled_train
combined_test = rolled_test
# combined_train = pd.concat([train_corr, rolled_train], axis=1)
# combined_train.columns = [ i for i in range(combined_train.shape[1])]
print('data-set is combined completely')

#
# tr_indi = [ np.arange((30*i),(30*i)+25) for i in range(198) ]
# tr_indi = np.concatenate(tr_indi)
# te_indi = [ np.arange((30*i)+25, (30*i)+30) for i in range(198)]
# te_indi = np.concatenate(te_indi)
# sub_indi = [ np.arange((30*i)+25,(30*i)+30) for i in range(1548-828)]
# sub_indi = np.concatenate(sub_indi)

#
combined_train['label'] = label_list
combined_train = combined_train.sample(frac=1).groupby('label').head(30)
# combined_test = combined_test.loc[sub_indi]



X_train, X_test, Y_train, Y_test = train_test_split(combined_train.iloc[:,:-1],combined_train['label'], random_state=1206, test_size=0.3)
# model = RandomForestClassifier(n_estimators=300, n_jobs=-1,random_state=1206)
# model.fit(combined_train.iloc[:,:-1], combined_train['label'])
# pred = model.predict_proba(combined_test)

# def step_decay(boosting_round):
# 	initial_lrate = 0.1
# 	drop = 0.5
# 	n_drop = 10.0
# 	lrate = initial_lrate * math.pow(drop, math.floor((1+boosting_round)/n_drop))
# 	return lrate

# lrate_list = [step_decay(i) for i in range(1,51)]
# , callbacks =[xgb.callback.reset_learning_rate(lrate_list)]

model = xgb.XGBClassifier(n_jobs=-1, objective='multi:softprob', n_estimators=100,
    max_depth=9, learning_rate=0.1)
eval_set=[(X_train, Y_train), (X_test, Y_test)]
model.fit(X_train,Y_train , eval_metric=["merror","mlogloss"], eval_set=eval_set, verbose=True)

#%%
# params = {'boosting_type':'goss',
#     'learning_rate': 0.01,
#     'max_depth': 100,
#     'num_leaves': 256,
#     'metric':'multi_logloss',
#     # 'feature_fraction': 0.1,
#     # 'bagging_fraction': 0.5,
#     # 'max_bin': 255}
#     'n_estimators':100,
#     'n_jobs':-1}
# model = LGBMClassifier(boosting_type='gbdt',
#          learning_rate = 0.1,
#           max_depth = -1,
#           num_leaves = 256,
#           n_estimators = 100,
#           n_jobs = -1)
# eval_set=[(X_train, Y_train), (X_test, Y_test)]
# model.fit(X_train,Y_train ,eval_metric="logloss", eval_set=eval_set, verbose=True)

#%%

filename = 'model/v1.sav'
joblib.dump(model, filename)
model.save_model('model/v1_backup')

# model.fit(X_train,Y_train)
# pred = model.predict_proba(X_test)
# pred = pd.DataFrame(pred)
# print(pred.max(axis=1))
# print(log_loss(Y_test,pred))

#%%
pred = model.predict_proba(combined_test)
submission = pd.DataFrame(pred)
submission.index= np.repeat(test_files,5)
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('output/submission13.csv', index=True)
print('****the process is done****')




