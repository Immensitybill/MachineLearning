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


def gradient (thetas,X,Y,lamda):
    X0 = X['F00']
    theta0 = thetas[0]
    a0 = hypothesis(theta0, X0) - Y
    gradient0 = np.dot(X0.T, a0) / len(X0)


    X1 = X.drop(['F00'],axis=1)
    theta1 = np.delete(thetas,0,axis=0)
    a1 = hypothesis(theta1,X1)-Y
    gradient1 = np.dot(X1.T,a1)/len(X1)+lamda/len(X1)*theta1

    result = np.insert(gradient1,0,values=gradient0,axis=0)
    return result