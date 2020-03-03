#%%
import numpy as np
import pandas as pd
from module.point_detector_v1 import feature_detector
import os
import glob
from functools import partial
import multiprocessing
from multiprocessing import Pool
from module.data_loader import data_loader
import warnings 
warnings.filterwarnings('ignore')
# %%
train_path = 'data/train/'
test_path = 'data/test/'
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
label = pd.read_csv('data/train_label.csv')

# %%
def data_loader_all(func, files, folder='', train_label=None, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df

# %%
train = data_loader_all(data_loader, train_list, folder=train_path, train_label=label, nrows=60)
#%%
c
#%%
train = train.loc[:,train.nunique()!=1]

#%%
conti_col = train.columns[ttrain.nunique()>9]
conti_col = set(conti_col) - set(['time','label'])
conti_col = list(conti_col)
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled =scaler.fit_transform(train[conti_col])

#%%
train[conti_col] = scaled

#%%
ttest = train.loc[train['time']<25]
test = test.loc[test['time']>54]
ttrain = train.loc[train['time']>19]
#%% shuffle and choose 45 sample from each group
ttrain = ttrain.sample(frac=1).groupby('label').head(40)
ttrain = ttrain.loc[:,ttrain.nunique()!=1]
#%%
col = list(ttrain.columns)
col.remove('time')
col.remove('label')

#%%
X_train = ttrain[col]
Y_train = ttrain['label']
X_test = test[col]
# Y_test = ttest['label']
#%%
corr_train = np.swapaxes(corr_train,0,2)


#%%    
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=500, n_jobs=-1,random_state=1206)

model1.fit(X_train, Y_train)
pred1 = model1.predict_proba(X_test)
#%%
submission = pd.DataFrame(pred1)
submission.index = test.loc[test['time']>39].index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('output/submission07.csv', index=True) 
#%%

#%% XGB
import xgboost as xgb
model2 = xgb.XGBClassifier(n_jobs=-1, objective='multi:softmax', n_estimators=10)
model2.fit(X_train,Y_train)
pred2 = model2.predict_proba(X_test)

#%% LGB
import lightgbm as lgb

model3 = lgb.LGBMClassifier(objective='multiclass', num_class=198, metric='multi_logloss',
                             n_estimators=10, n_jobs=-1)
model3.fit(X_train,Y_train)
pred3 = model3.predict(X_test)
#%%
params = {
          "objective" : "multiclass",
          "num_class" : 198,
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1 }


lgtrain, lgval = lgb.Dataset(X_train, Y_train), lgb.Dataset(X_test, Y_test)
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=200)

#%% knn
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier(n_neighbors=198, n_jobs=-1)
model4.fit(X_train, Y_train)
pred4 = model4.predict_proba(X_test)



#%%
import catboost as cab
model5 = cab.CatBoostClassifier(
                               loss_function='MultiClass',
    #                            eval_metric="AUC",
                               task_type="CPU",
                               learning_rate=0.01,
                               iterations=10,
                               od_type="Iter",
#                                depth=8,
                              
    #                            l2_leaf_reg=1,
    #                            border_count=96,
                               random_seed=1206
                              )

model5.fit(X_train,Y_train)
pred5 = model5.predict_proba(X_test)


#%%
from sklearn.metrics import classification_report
print(classification_report(Y_test.ravel()[24::25], pred1[24::25]))


#%%
uniq_label = np.where(train.groupby('label').count()['time']==60)[0] # true index
uniq_train = train.loc[train['label'].isin(uniq_label)]
#%%


#%%
cate_col = train.columns[train.nunique()<10]


#%% variant -> corr
glob_set = set(train.columns)
for i in range(828):
    col_set = set(train.columns[train.iloc[60*i:60*(i+1),:].nunique()>1])
    glob_set = glob_set.intersection(col_set)

#%%


#%% 
comb=[]
for i in range(len(glob_set)):
    if i - 0 >0 :
        temp = [[j,i] for j in range(i)]
        comb.append(temp)

#%%
from itertools import chain
col_comb = list(chain.from_iterable(comb))
col_comb
#%%


#%%
pd.DataFrame(a['c'].ravel())
#%%
def corr_generator(flist):
    data = train.loc[flist]
    for i in range(45):
        tcorr = np.array(data[col].iloc[i:i+15,:].corr())
        if i == 0:
            table = tcorr
        else :
            table = np.dstack([table,tcorr])
 
    table = table.reshape(len(col)*len(col),45)
    table = pd.DataFrame(table).drop_duplicates().values
    return table
#%%

#%%

#%%
from functools import partial
import multiprocessing
from multiprocessing import Pool
def corr_merger(flist):   
    func_fixed = partial(corr_generator)     
    
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        corr_list = list(pool.imap(func_fixed, flist)) 
        pool.close()
        pool.join()        
    combined_corr = np.dstack(corr_list)    
    return combined_corr

#%%

import random
col = random.sample(list(train.columns),200)
files = np.arange(828)
corr_train = corr_merger(files)
corr_train = np.swapaxes(corr_train,0,2)

#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Activation
#%%

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(198))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
#%%
# tr_indi = [ np.arange((45*i)+20,(45*i)+60) for i in range(828) ]
# tr_indi = np.concatenate(tr_indi)
# te_indi = [ np.arange((45*i)+15, (45*i)+20) for i in range(828)]
# te_indi = np.concatenate(te_indi)

#%%
label_list = []
for i in range(828):
    label_list.append(train['label'].loc[i].iloc[0])
label_list = np.repeat(label_list, 45)
#%%
X = corr_train.reshape(37260,200,200,1)
Y = label_list

#%%
model.fit(X,Y ,batch_size=100, epochs=5, validation_split = 0.1)
# %%
same_col = train[train['time']==0].nunique()==1
diff_col = train[train['time']==0].nunique()!=1

# %%
same_df = train.loc[:,list(same_col)]
diff_df = train.loc[:,list(diff_col)]
same_df['label'] = train['label']
diff_df['time'] = train['time']

# %%
same_df = same_df.loc[:,same_df.nunique()!=1]

# %%

# %%
train = train.loc[:,train.nunique()!=1]
# %% dynamic corr
for j in range(198): # id
    a = train.loc[j]
    b = np.array([a.columns])
    for i in range(45): # sliding window
        corr = a.iloc[i:i+15,:].corr()
        b = np.vstack((b,corr.iloc[1,:].values))
    
    if j ==0:
        c = b.copy()
    c = np.dstack((c,b))

# %%

# %%


# %%
45*828

# %%
