import numpy as np
from matplotlib import pyplot as plt

class GradientDescenter:
    @staticmethod
    def computeCostFunction(trainingDatas,labels,theta):

        m,n = np.shape(trainingDatas)
        loss = np.dot(trainingDatas,theta,)-labels
        squareCost = np.power(loss,2)

        # sum = np.sum(squareCost)
        #
        # result1 = sum/(2*len(trainingDatas))

        result = sum(squareCost)/(2*m)

        # result = np.dot(squareCost,np.ones(np.shape(squareCost)[0]))/(2*m)
        return result

    @staticmethod
    def batchGradientDescent(x,y,alpha,iterations):
        m,n = np.shape(x)
        theta = np.zeros(n)
        xtrans = np.transpose(x)
        costs = []
        for i in range(0,iterations):
            loss = np.dot(x,theta,)-y
            theta = theta - alpha/m*(np.dot(xtrans,loss))
            cost = GradientDescenter.computeCostFunction(x,y,theta)
            costs.append(cost)
            print(cost)
        print (theta)
        xAxis = np.arange(0, iterations)
        plt.plot(xAxis, costs)
        plt.show()
        return theta