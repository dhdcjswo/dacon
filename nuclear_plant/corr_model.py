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

train_corr =pd.read_pickle('preprocess/train_corr.pkl')
train_corr = train_corr.fillna(0)

test_corr = pd.read_pickle('preprocess/test_corr.pkl')
test_corr = test_corr.fillna(0)
#%%
col = train_corr.std().sort_values(ascending=False).head(1000).index
Y = np.repeat(label['label'],30)


#%%

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_corr[col], Y, test_size=0.3)
# %%

model = xgb.XGBClassifier(n_jobs=-1, objective='multi:softprob', n_estimators=50,
    max_depth=5, learning_rate=0.1)
eval_set=[(X_train, Y_train), (X_test, Y_test)]
model.fit(X_train,Y_train , eval_metric=["merror","mlogloss"], eval_set=eval_set, verbose=True)


# %% 1154, 1168
label = np.repeat(np.arange(828,1154),30)
label = np.append(label, np.repeat(1154,4))
label = np.append(label, np.repeat(np.arange(1155,1168),30))
label = np.append(label, np.repeat(1168,4))
label = np.append(label, np.repeat(np.arange(1169,1548),30))
#%%
pred = model.predict_proba(test_corr[col])
submission_corr = pd.DataFrame(pred)
submission_corr.index = label
submission_corr.index.name = 'id'
submission_corr = submission_corr.groupby('id').mean()
submission_corr.to_csv('model/stacking/submission_corr.csv', index=True)
#%%
sub3.idxmax(axis=1)

# %%
sub1= pd.read_csv('model/stacking/submission_rolled.csv')
sub2 = pd.read_csv('model/stacking/submission_point.csv')
sub3 = submission_corr.copy()
sub3 = sub3.reset_index()
#%%

# %%
submission = sub2.copy()
submission.iloc[:,1:] = (sub2.iloc[:,1:]*0.5)+(sub3.iloc[:,1:]*0.5).values

# %%
submission.to_csv('model/stacking/submission.csv', index=False)
