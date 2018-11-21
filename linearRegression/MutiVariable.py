import pandas as pd
import numpy as np


from numpy import genfromtxt

from linearRegression import DataGetter, OneVariable

path =  'F:\PyWorkSpace\machine learning\linearRegression\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
dataScaled = (data-data.mean())/data.std()
X = dataScaled.loc[:,'Size':'Bedrooms']
Y = dataScaled.loc[:,'Price']
X.insert(0, 'Ones', 1)

m,n = np.shape(X)

theta = np.zeros(n)

OneVariable.batchGradientDescent(X,Y,1,10)