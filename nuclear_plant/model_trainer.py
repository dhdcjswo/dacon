#%%
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss
import dask.dataframe as dd
#%%
label_list = pd.read_csv('preprocess/label_list.csv' )
corr_train = dd.read_csv('preprocess/corr_train.csv' ) 
corr_test = dd.read_csv('preprocess/corr_test.csv')
label_list = label_list.iloc[:,0].values
#%%
corr_train['label'] = label_list
corr_train = corr_train.sample(frac=1).groupby('label').head(30)
#%%
corr_train = corr_train.sort_values('label')
corr_train = corr_train.reset_index(drop=True)
#%%

#%%
corr_train = corr_train.compute()
corr_test = corr_test.compute()
#%%
corr_train = corr_train.reset_index(drop=True)
corr_test = corr_test.reset_index(drop=True)
#%%
tr_indi = [ np.arange((30*i),(30*i)+25) for i in range(198) ]
tr_indi = np.concatenate(tr_indi)
te_indi = [ np.arange((30*i)+25, (30*i)+30) for i in range(198)]
te_indi = np.concatenate(te_indi)
sub_indi = [ np.arange((30*i)+20,(30*i)+30) for i in range(1548-828)]
sub_indi = np.concatenate(sub_indi)

#%%
X_train = corr_train.loc[tr_indi].fillna(0)
Y_train = label_list[tr_indi]
X_test = corr_train.loc[te_indi].fillna(0)
Y_test = label_list[te_indi]
sub_test = corr_test.loc[sub_indi].fillna(0)
# %%
model = RandomForestClassifier(n_estimators=300, n_jobs=-1,random_state=1206, )

model.fit(X_train, Y_train)
pred1 = model.predict(X_test)
pred2 = model.predict_proba(X_test)
# %%
# print(log_loss(Y_test, pred1))
print(classification_report(Y_test,pred1))
# %%
pred = model.predict_proba(sub_test)
submission = pd.DataFrame(pred)
id_list = np.repeat(np.arange(828,1548),10)
submission['id'] = id_list
submission = submission.set_index('id')
submission = submission.groupby('id').mean()
#%%
submission.to_csv('output/submission09.csv', index=True) 


#%%
filename = 'model/corr_model1_sub9.sav'
joblib.dump(model, filename)

# %%
pd.DataFrame(pred2).max(axis=1).head(50)

# %%
import xgboost as xgb
model2 = xgb.XGBClassifier(n_jobs=-1, objective='multi:softmax', n_estimators=100)
eval_set=[(X_train, Y_train), (X_test, Y_test)]
model2.fit(X_train,Y_train , eval_metric=[“auc”,“logloss”], eval_set=eval_set, verbose=True)
# pred2 = model2.predict_proba(X_test)
# pred1 = model2.predict(X_test)

# %%


# %%
import lightgbm as gbm

mod = gbm.LGBMClassifier()
train_lgb = gbm.Dataset(data = corr_train,label = label_list)






N_FOLDS = 3
MAX_EVALS=5

cv_results = gbm.cv(default, train_lgb, num_boost_round = 1000, early_stopping_rounds = 100, 
                   metrics = 'multi_logloss',  nfold = N_FOLDS, seed = 42)
cv_results



# %%
model = gbm.LGBMClassifier()
model.fit(X_train, Y_train)
#%%
corr_train.head()
# %%
from sklearn.decomposition import PCA

# X = pd.concat([X_train,X_test])
X = corr_train.compute().fillna(0)
pca1 = PCA(n_components=100)
X_low = pca1.fit_transform(X.iloc[:,:-2])

X_train1 = X_low[:len(X_train)]
X_test1 = X_low[len(X_train):]

# %%
