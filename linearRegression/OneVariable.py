from linearRegression import DataGetter
import numpy as np
from matplotlib import pyplot as plt

# dataPath = "F:\PyWorkSpace\machine learning\linearRegression\ex1data1.txt"
# trainingDatas, labels = DataGetter.getData(dataPath)
# m, n = np.shape(trainingDatas)

def computeCostFunction(trainingDatas,labels,theta):

    m,n = np.shape(trainingDatas)
    # theta = np.ones(n)
    loss = np.dot(trainingDatas,theta,)-labels
    squareCost = np.power(loss,2)

    # sum = np.sum(squareCost)
    #
    # result1 = sum/(2*len(trainingDatas))

    result = sum(squareCost)/(2*m)

    # result = np.dot(squareCost,np.ones(np.shape(squareCost)[0]))/(2*m)
    return result


# computeCostFunction(trainingDatas,labels,np.zeros(n))


def batchGradientDescent(x,y,alpha,iterations):
    m,n = np.shape(x)
    theta = np.zeros(n)
    xtrans = np.transpose(x)
    costs = []
    for i in range(0,iterations):
        loss = np.dot(x,theta,)-y
        theta = theta - alpha/m*(np.dot(xtrans,loss))
        cost = computeCostFunction(x,y,theta)
        costs.append(cost)
        print(cost)
    print (theta)
    xAxis = np.arange(0, iterations)
    plt.plot(xAxis, costs)
    plt.show()
    return theta


# theta = batchGradientDescent(trainingDatas,labels,0.01,1000)
#
#
# population = trainingDatas[:,:-1]
# x = np.linspace(population.min(),population.max(),100)
# y = theta[0]*x + theta[1]
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, y,'r',label='Prediction')
# ax.scatter(trainingDatas[:,0],labels,label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()