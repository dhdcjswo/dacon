#%%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


label = pd.read_csv('data/train_label.csv')
train =pd.read_pickle('preprocess/train_over.pkl')
train = train.fillna(0)
test = pd.read_pickle('preprocess/test_over.pkl')
test = test.fillna(0)

train = train.loc[:,train.nunique()>9]
test = test[train.columns[:-1]]


#%%
rolled_train = train.iloc[:,1:].copy()
rolled_train = rolled_train.groupby(train.index).rolling(15).mean()
rolled_train = rolled_train.dropna()
rolled_train = rolled_train.sample(frac=1).groupby('label').head(150)
rolled_train = rolled_train.reset_index(drop=True)

rolled_test = test.iloc[:,1:].copy()
rolled_test = rolled_test.groupby(test.index).rolling(15).mean()
rolled_test = rolled_test.dropna()
rolled_test = rolled_test.groupby(rolled_test.index).tail(30)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(rolled_train.iloc[:,:-1], rolled_train['label'], test_size=0.3)

import xgboost as xgb
model = xgb.XGBClassifier(n_jobs=-1,objective='multi:softprob', n_estimators=50,
    max_depth=5, learning_rate=0.1)
eval_set=[(X_train, Y_train), (X_test, Y_test)]
model.fit(X_train,Y_train , eval_metric=["merror","mlogloss"], eval_set=eval_set, verbose=True)

model.save_model('model/stacking/rolled_model')

pred= model.predict_proba(rolled_test)
submission_rolled = pd.DataFrame(pred)
submission_rolled.index = np.repeat(np.arange(828,1548),30)
submission_rolled.index.name = 'id'
submission_rolled = submission_rolled.groupby('id').mean()
submission_rolled.to_csv('/content/drive/My Drive/dacon/plant/submission_rolled.csv', index=True)
