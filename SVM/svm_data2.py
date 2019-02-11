import scipy.io as io
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


mat = io.loadmat('./data/ex6data2.mat')
data = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
data['y'] = mat.get('y')

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'],data['X2'],s=50, c=data['y'], cmap='RdBu',alpha=0.5 )
plt.show()
svc = SVC(kernel='rbf',gamma=10,C=100, probability=True)
svc.fit(data[['X1','X2']],data['y'])
score = svc.score(data[['X1','X2']],data['y'])


data['problity'] = svc.predict_proba(data[['X1','X2']])[:,1]
data['predict'] = svc.predict(data[['X1','X2']])

# data['problity'] = svc.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'],data['X2'],s=50, c=data['problity'], cmap='RdBu',alpha=0.5 )
plt.show()

print(score)