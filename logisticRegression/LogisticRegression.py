import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# dataPath = "F:\PyWorkSpace\machine learning\logisticRegression\ex2data1.txt"
# data = pd.read_csv(dataPath, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#
# exam12 = data[['Exam 1','Exam 2']]
# admitted = data['Admitted']
# m,n = np.shape(exam12)
# thetas = np.zeros(n)

def sigmoid(z):
    return 1/(1+np.exp(-z))



def hypothesis(thetas, X):
    z = np.dot(X,thetas)
    result = sigmoid(z)
    return result


# print(hypothesis(exam12,thetas))


def costFunction(thetas,X,Y):

    firstPart = Y*np.log(hypothesis(thetas, X))
    secondPart = (1-Y)*np.log(1 - hypothesis(thetas, X))
    m = len(X)
    result = np.sum(firstPart+secondPart)*(-1/m)
    return result


def gradient (thetas,X,Y,):
    a = hypothesis(thetas, X) - Y
    gradient = np.dot(X.T, a) / len(X)
    return gradient

def gradientDescent(X,Y,thetas,alpha,maxIterations):
    costs = []
    for i in range(0,maxIterations):
        print('i: ', i)
        a = hypothesis(thetas, X) - Y
        gradient = np.dot(X.T,a)/len(X)

        # gradient =(1 / len(X)) * X.T @ (sigmoid(X @ thetas) - Y)
        print('gradient: ',gradient)
        cost = costFunction(thetas,X,Y)
        costs.append(cost)
        print('cost: ',cost)
        print('old thetas: ',thetas)
        thetas = thetas - alpha*gradient
        print('new thetas: ', thetas)
        print('hyposthesis: ', hypothesis(thetas, X))

        # costFunction(X,Y,thetas)
    xAxis = np.arange(0, maxIterations)
    plt.plot(xAxis, costs)
    plt.show()
    print('final thetas: ',thetas)
    return thetas


# gradientDescent(exam12,admitted,thetas,0.01,400)


# x = np.arange(-10,10)
# y = sigmoid(x)
#
# plt.plot(x,y),
#
# plt.show()

