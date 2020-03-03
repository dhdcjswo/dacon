#%% kmenas elbow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#%%
data = scaler.fit_transform(np.array(train.iloc[:,1:-1].mean()).reshape(-1,1))
# X_train = data[:len(X_train)]
# X_test = data[len(X_test):]

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(10,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



#%%
kmeanModel = KMeans(n_clusters=6)
kmeanModel.fit(data)
clusters = kmeanModel.predict(data)
#%%
plt.hist(clusters)

#%% PCA
#%%
from sklearn.decomposition import PCA,KernelPCA

X = pd.concat([X_train,X_test])
pca1 = PCA(n_components=100)
X_low = pca1.fit_transform(X.iloc[:,:-2])

X_train1 = X_low[:len(X_train)]
X_test1 = X_low[len(X_train):]
