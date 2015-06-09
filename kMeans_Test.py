import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv('Test_Data_3.csv')

df.head()

for x in df[1::]:
	float(x)

dfTrain, dfTest = train_test_split(df, test_size=0.2)

cl = dfTrain[:,16]
true_labels = dfTest[:,16]

from sklearn.cluster import KMeans

model = KMeans(n_clusters=20, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)
model.fit(dfTrain[:,:16], dfTrain[:,16])
predicted = model.predict([750,0,1,150,60,35,300000,70,8.5,200,50,58000,150,1,10,10])
print predicted
