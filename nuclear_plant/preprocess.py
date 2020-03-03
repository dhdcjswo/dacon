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
def data_loader_all(func, files, folder='', train_label=None, nrows=45, skiprows = range(1,16)):   
    func_fixed = partial(func, folder=folder, train_label=train_label, nrows=nrows, skiprows=skiprows)
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df


# %%
train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label, nrows=45, skiprows = range(1,16))
# test = data_loader_all(data_loader, test_list, folder=test_path, train_label=None, nrows=45, skiprows=range(1,16))

train = train.loc[:,train.nunique()>1]
#%%
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

#%%
data = scaler.fit_transform(pd.concat([train.iloc[:,1:-1],test.iloc[:,1:]], join='inner'))
#%%
kmeanModel = KMeans(n_clusters=30)
kmeanModel.fit(data)
#%%
train['cl'] = kmeanModel.labels_[:len(train)]
test['cl'] = kmeanModel.labels_[len(train):]

#%%
col = set(train.columns) - set(['time','label'])
#%%
a = train.sample(frac=1).groupby('label').head(45)
#%%
a.groupby('label').head(40)
a.groupby('label').tail(5)
#%%
X_train = a.groupby('label').head(40)[col]
Y_train = a.groupby('label').head(40)['label'] 
X_test  = a.groupby('label').tail(5)[col]
Y_test  = a.groupby('label').tail(5)['label'] 
#%%
X_train = train.groupby('label').head(45)[col]
Y_train = train.groupby('label').head(45)['label']
X_test = test[col]
# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, n_jobs=-1,random_state=1206)

model.fit(X_train, Y_train)
#%%
pred1 = model.predict(X_test)
pred2 = model.predict_proba(X_test)
# %%
from sklearn.metrics import log_loss, classification_report

print(classification_report(Y_test,pred1))
print(log_loss(Y_test,pred2))

# %%
submission = pd.DataFrame(pred2)
submission.index = test.loc[test['time']>49].index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('output/submission10.csv', index=True) 


# %%
mean_train = train.groupby(train.index).tail(45)
mean_test = test.groupby(test.index).tail(45)
mean_train.dropna().to_csv('preprocess/rolling_train.csv')
mean_test.dropna().to_csv('preprocess/rolling_test.csv')
