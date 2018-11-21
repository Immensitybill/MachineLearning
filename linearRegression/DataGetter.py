import numpy as np

from numpy import genfromtxt


def getData(dataPath):
    dataSet = genfromtxt(dataPath, delimiter=',')
    (m,n) = np.shape(dataSet)
    trainingData = np.ones((m,n))
    trainingData[:,:-1] =  dataSet[:,:-1]
    label= dataSet[:,-1]
    return trainingData, label
