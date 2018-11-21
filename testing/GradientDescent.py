import numpy as np
import random
from numpy import genfromtxt
from matplotlib import pyplot as plt

def getData(dataSet):
    m, n = np.shape(dataSet) #查看数组维数
    trainData = np.ones((m, n)) #创建一个m*n的元素都是1的数组
    trainData[:,:-1] = dataSet[:,:-1] #把dataSet里最后一列去掉，赋值给trainData,给training data加一个x0
    trainLabel = dataSet[:,-1] #拿dataSet的最后一列，作为label
    return trainData, trainLabel

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    result = []
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        lossSquare = loss * loss
        # print loss
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
        ones = np.ones((10,1))
        np.vdot(lossSquare,ones)
        result.append(np.vdot(lossSquare,ones)/(2*m))
        print("step",i,": ",np.vdot(lossSquare,ones)/(2*m))
    xAxis = np.arange(0,maxIterations)
    plt.plot(xAxis,result)
    plt.show()

    return theta

def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP

dataPath = r"F:\PyWorkSpace\test\data\test.csv"
dataSet = genfromtxt(dataPath, delimiter=',')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.1
maxIteration = 1000
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print(predict(x, theta))
