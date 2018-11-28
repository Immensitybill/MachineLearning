import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))


def hypothesis(theta, X):
    z = np.dot(X,theta)
    result = sigmoid(z)
    return result

def cost(X,y,output):
    firstPart = -y * np.log(output)
    secondPart = (1 - y) * np.log(1 - output)
    m = len(X)
    resultMarix = firstPart-secondPart
    sum_ones = np.ones(np.shape(resultMarix)[1])
    sum_k = np.dot(resultMarix,sum_ones)
    return np.sum(sum_k)/m