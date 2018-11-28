import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def hypothesis(thetas, X):
    z = np.dot(X,thetas)
    result = sigmoid(z)
    return result

def costFunction(thetas,X,Y,lamda):

    firstPart = Y*np.log(hypothesis(thetas, X))
    secondPart = (1-Y)*np.log(1 - hypothesis(thetas, X))
    m = len(X)

    regularization = (lamda/(2*m))*np.sum(np.power(thetas,2))
    result = np.sum(firstPart+secondPart)*(-1/m) + regularization
    return result

def gradient (thetas,X,Y,):
    a = hypothesis(thetas, X) - Y
    gradient = np.dot(X.T, a) / len(X)
    return gradient

def regularized_gradient(theta, X, y, l):
#     '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    g = gradient(theta, X, y)
    result = g + regularized_term
    return result


# def gradient(theta, X, y, learningRate):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#
#     parameters = int(theta.ravel().shape[1])
#     error = sigmoid(X * theta.T) - y
#
#     grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
#
#     # intercept gradient is not regularized
#     grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
#
#     return np.array(grad).ravel()

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg