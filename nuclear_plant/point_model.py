#%%
import numpy as np
import pandas as pd
import random
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
import pickle

label = pd.read_csv('data/train_label.csv')
train =pd.read_pickle('preprocess/train_over.pkl')
train = train.fillna(0)
test = pd.read_pickle('preprocess/test_over.pkl')
test = test.fillna(0)
#%%
train = train.loc[:, train.nunique()>9]
test = test[train.columns[:-1]]
#%%

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(pd.concat([train.iloc[:,1:-1], test.iloc[:,1:]]))
train_v = train.copy()
test_v = test.copy()
#%%
train_v.iloc[:,1:-1] = scaled[:len(train)]
test_v.iloc[:,1:] = scaled[len(train):]
col = train_v.iloc[:,1:-2].std().sort_values(ascending=False).head(200).index
#%%
col_sum = train_v.loc[train_v['time']==55][col].sum(axis=1)
train_v['sum'] = col_sum

col_sum = test_v.loc[test_v['time']==55][col].sum(axis=1)
test_v['sum'] = col_sum
#%%
train_v = train_v.sample(frac=1).groupby('label').head(250)
# %%
col = set(train_v.columns) - set(['time','label'])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_v[col], train_v['label'], test_size=0.3)
# %%
model = xgb.XGBClassifier(n_jobs=-1, objective='multi:softprob', n_estimators=30,
    max_depth=5, learning_rate=0.1)
eval_set=[(X_train, Y_train), (X_test, Y_test)]
model.fit(X_train,Y_train , eval_metric=["merror","mlogloss"], eval_set=eval_set, verbose=True)

# %%
model.save_model('model/stacking/xgb_model_point')
#%%
pred=model.predict_proba(test_v[col])

# %%
submission_point = pd.DataFrame(pred)
submission_point.index = list(test_v.index)
submission_point.index.name = 'id'
submission_point = submission_point.sort_index()
submission_point = submission_point.groupby('id').mean()
submission_point.to_csv('model/stacking/submission_point.csv',index=True)
# %%


# %%
