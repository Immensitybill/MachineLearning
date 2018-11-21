import numpy as np

import matplotlib.pyplot as plt
from numpy import genfromtxt

dataPath = "F:\PyWorkSpace\machine learning\linearRegression\ex1data1.txt"
dataSet = genfromtxt(dataPath, delimiter=',')

x=dataSet[:,0]
y=dataSet[:,1]

plt.scatter(x,y,marker='x',c='red')
plt.grid(True)
plt.show()