import numpy as np
import pandas as pd
import scipy.optimize as opt

from logisticRegression import LogisticGradientDescenter

dataPath = "F:\PyWorkSpace\machine learning\logisticRegression\ex2data2.txt"
data = pd.read_csv(dataPath, header=None, names=['test1', 'test2', 'Admitted'])

test12 = data[['test1','test2']]





def mapFeature(X, degree):
    degree=degree+1
    x1 = X['test1']
    x2 = X['test2']
    for i in range(0, degree):
        for j in range (0 , degree-i):
            X ['F'+ str(i)+str(j)] = np.power(x1,i)*np.power(x2,j)
    X.drop('test1',axis=1,inplace=True)
    X.drop('test2',axis=1,inplace=True)
    return X


X = mapFeature(test12,6)
Y = data['Admitted']
m,n = np.shape(X)
thetas = np.zeros(n)

result = opt.fmin_tnc(func=LogisticGradientDescenter.costFunction, x0=thetas, fprime=LogisticGradientDescenter.gradient, args=(X, Y))

print(result)

hy = LogisticGradientDescenter.hypothesis(result[0],X)

result = []
for i in hy:
    if i >= 0.5:
        result.append(1)
    else:
        result.append(0)

sum = 0
for j in range(0,len(result)):
    if result[j] == Y[j]:
        sum = sum+1

print ("correct ratio: ",sum/len(result))



