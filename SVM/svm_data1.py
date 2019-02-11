import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
import numpy as np

mat = io.loadmat('./data/ex6data1.mat')
data = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
data['y'] = mat.get('y')

# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(data['X1'],data['X2'],s=50, c=data['y'], cmap='RdBu',alpha=0.5 )
# plt.show()

svc1 = svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1','X2']],data['y'])
svc1.score(data[['X1','X2']],data['y'])

data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

# fig,ax= plt.subplots()
# ax.scatter(data["X1"], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
# plt.show()

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum(((x1-x2) ** 2))/(2 * sigma ** 2))

print(gaussian_kernel(np.array([1,2,1]),np.array([0,4,-1]),2))

