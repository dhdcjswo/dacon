#%%
import numpy as np
import pandas as pd
import os
import glob
from functools import partial
import multiprocessing
from multiprocessing import Pool
from module.data_loader import data_loader

import warnings 
import random
import joblib
warnings.filterwarnings('ignore')
# %%
train_path = 'data/train/'
test_path = 'data/test/'
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
label = pd.read_csv('data/train_label.csv')

# %%
def data_loader_all(func, files, folder='', train_label=None,  skiprows = range(1,16)):   
    func_fixed = partial(func, folder=folder, train_label=train_label, skiprows=skiprows)
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df

train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label, skiprows = range(1,16))
# test = data_loader_all(data_loader, test_list, folder=test_path, train_label=None, nrows=45, skiprows=range(1,16))
#%%
train.to_pickle('train')
#%%
# def data_loader_all(func, files, folder='', train_label=None, nrows=60):   
#     func_fixed = partial(func, folder=folder, train_label=train_label, nrows=nrows)
#     if __name__ == '__main__':
#         pool = Pool(processes=multiprocessing.cpu_count()) 
#         df_list = list(pool.imap(func_fixed, files)) 
#         pool.close()
#         pool.join()        
#     combined_df = pd.concat(df_list)    
#     return combined_df

# train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label, nrows=60)
# test = data_loader_all(data_loader, test_list, folder=test_path, train_label=None, nrows=60)
#%%
train = train.loc[:,train.nunique()>1]
#%%
train.shape
#%%
import time
start_time = int(time.time())
bef_corr = bef.corr()
aft_corr = aft.corr()
result = (aft_corr - bef_corr.values).abs()
result = result.fillna(0)
print('Seconds: %s' % (time.time() - start_time))
#%%
mask = np.ones((2374,2374),dtype='bool')
mask[np.triu_indices(2374)] = False

#%%
# ref2 = result[(result>2) & mask].stack().index.tolist()
ref1 = result[(result>1) & mask].stack().index.tolist()
# ref0 = result[(result>1) & mask].stack().index.tolist()
#%%
for i in range(828):
    if i != 30 :
        data = train.loc[i]
        data = data.loc[:,data.nunique()>10]
        mask = np.ones((data.shape[1]-2,data.shape[1]-2),dtype='bool')
        mask[np.triu_indices(data.shape[1]-2)] = False
        bef = data.loc[data['time']<15]
        aft = data.loc[data['time']>14]
        bef = bef.iloc[:,1:-1]
        aft = aft.iloc[:,1:-1]
        bef_corr = bef.corr()
        aft_corr = aft.corr()
        result = (aft_corr - bef_corr.values).abs()
        result = result.fillna(0)
        result = result[(result>1) & mask].stack().index.tolist()
        if i == 0:
            comb_corr = set(result)
        else :
            comb_corr = comb_corr | set(result)
    else :
        pass
#%%
comb_corr = pd.DataFrame(comb_corr)
comb_corr.to_csv('preprocess/comb_corr_2.csv', index=False)
#%%
comb_corr = pd.read_csv('preprocess/comb_corr_2.csv')
comb_corr = comb_corr.values
#%%

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(train.iloc[:,1:-1])
#%%
ma = np.abs(np.max(scaled, axis=0))
mi = np.abs(np.min(scaled, axis=0))
#%%
ma = np.where(ma>2)
mi = np.where(mi>2)
#%%
col_list = ma or mi
col_list = col_list[0]
#%%
col_list = random.sample(list(col_list),300)
col_list = pd.DataFrame(col_list)
col_list.to_csv('preprocess/col_list.csv', index=False)

#%%
col_list = pd.read_csv('preprocess/col_list.csv')
col = col_list.iloc[:,0].values

#%%

train.index
# %%
col = np.arange(len(train.columns)-2)
col = random.sample(list(col),200)

def column_generator():
    data = train.loc[0]

    for i in range(30):
        tcorr = np.array(data.iloc[i:i+15,col].corr())
        if i == 0:
            table = tcorr
        else :
            table = np.dstack([table,tcorr])
    
    table = table.reshape(len(col)*len(col),30)
    table = pd.DataFrame(table.astype('float16')).drop_duplicates()
    ind = table.index
    
    return list(ind)
#%%
corr_index = column_generator()
#%%
corr_inde = pd.DataFrame(corr_index)
corr_inde.to_csv('preprocess/corr_index.csv', index=False)
#%%
corr_index = pd.read_csv('preprocess/corr_index.csv')
corr_index = corr_index.iloc[:,0].values
#%%
train_files = np.arange(828) 
test_files = np.arange(828,1548)
    
#%%
mask = np.ones((len(col),len(col)),dtype='bool')
mask[np.triu_indices(len(col))] = False
def corr_generator(flist):
    data = train.loc[flist]
    
    for i in range(len(data)-15):
        tcorr = np.array(data.iloc[i:i+15,col].corr())[mask]
        if i == 0:
            table = tcorr
        else :
            table = np.vstack([table,tcorr])
    
    dlabel = label.loc[flist]['label'] 
    table = np.hstack([table,np.repeat(dlabel, len(table)).reshape(-1,1)])
    return table

#%%
def corr_merger(flist):   
    func_fixed = partial(corr_generator)     
    
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        corr_list = list(pool.imap(func_fixed, flist)) 
        pool.close()
        pool.join()        
    combined_corr = pd.concat(corr_list)    
    return combined_corr
#%%
import gc
gc.collect()

# %%
train_corr = corr_merger(train_files)
#%%
train_corr = train_corr.reset_index(drop=True)
train_corr.to_csv('preprocess/corr_train.csv', index=False)
#%%
from sklearn.decomposition import PCA,KernelPCA

pca1 = PCA(n_components=1000)
train_corr = pca1.fit_transform(train_corr)



#%%
test = corr_merger(test_files)
test = test.reset_index(drop=True)
test.to_csv('preprocess/corr_test.csv', index=False)
#%%

#%%

label_list = []
for i in range(828):
    label_list.append(train['label'].loc[i].iloc[0])
label_list = np.repeat(label_list, 30)
#%%
label_list = pd.DataFrame(label_list)
label_list.to_csv('preprocess/label_list.csv', index=False)

#%%
rolled_train = train.copy()
rolled_train = rolled_train.groupby(train.index).rolling(15).mean()
rolled_train = rolled_train.dropna()
rolled_train = rolled_train.groupby(rolled_train.index).tail(30)
rolled_train = rolled_train.reset_index(drop=True)
#%%
combined_train = pd.concat([train_corr,rolled_train.iloc[:,1:-1]], axis=1)
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(combined_train, label_list, random_state=1206, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=1206)
model.fit(X_train, Y_train)
pred1 = model.predict(X_test)
pred2 = model.predict_proba(X_test)
print(classification_report(Y_test,pred1))
print(log_loss(Y_test,pred2))
#%%
import matplotlib.pyplot as plt
plt.hist(np.max(pred2,axis=1))

#%%
col = set(train.columns) - set(['time','label'])
#%%
train = train.loc[train['time']>14]
X_train = train.loc[train['time']<50][col]
X_test  = train.loc[train['time']>49][col]

#%%
glob_loss =0
for i in range(10):
    
    label = pd.DataFrame({'label2':np.random.randint(0,2,198)})
    train = train.merge(label, left_on='label', right_on=label.index)
    Y_train = train.loc[train['time']<50]['label2']
    Y_test  = train.loc[train['time']>49]['label2']
    print('Model {} is running'.format(i))
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1,random_state=1206)
    model.fit(X_train, Y_train)
    pred1 = model.predict(X_test)
    loss = accuracy_score(Y_test, pred1)
    print(loss)
    if loss > glob_loss:
        glob_loss = loss
        best_split = label.iloc[:,0].values
    del train['label2']
    


# %%
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, classification_report, accuracy_score

pca1 = PCA(n_components=100)
# pca1 = KernelPCA(n_components=300, kernel='poly')
X_low = pca1.fit_transform(train.iloc[:,1:-1])
X_train, X_test, Y_train, Y_test = train_test_split(X_low,train['label'], random_state=1206)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=1206)
model.fit(X_train, Y_train)
pred1 = model.predict(X_test)
print(classification_report(Y_test,pred1))

#%%
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(train.iloc[:,1:-1])
clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled)
plt.hist(clustering.labels_)



#%%
train = train.loc[train['time']>14]
X_train = train.loc[train['time']<50][col]
Y_train = train.loc[train['time']<50]['label']
X_test  = train.loc[train['time']>49][col]
Y_test  = train.loc[train['time']>49]['label']
#%%
# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300, n_jobs=-1,random_state=1206)
model.fit(X_train, Y_train)
#%%
pred1 = model.predict(X_test)
pred2 = model.predict_proba(X_test)
# %%
from sklearn.metrics import log_loss, classification_report, accuracy_score

print(classification_report(Y_test,pred1))
print(log_loss(Y_test,pred2))

# %%
submission = pd.DataFrame(pred2)
submission.index = test.loc[test['time']>49].index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('output/submission10.csv', index=True) 

#%%


import lightgbm as gbm

train_lgb = gbm.Dataset(data = X_low,label =train['label'] )
params = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 40, 
    'objective': 'multiclass',
    'num_class':198,
    'tree_learner':'voting',
    'metric':'multi_logloss',
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'max_bin': 100}


N_FOLDS = 3
MAX_EVALS=5
cv_results = gbm.cv(params, train_lgb, num_boost_round = 1000, early_stopping_rounds = 100, 
                   metrics = 'multi_logloss',  nfold = N_FOLDS, seed = 42)
cv_results

# %%


def step_decay(n_estimators):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+n_estimators)/epochs_drop))
	return lrate


#%%
