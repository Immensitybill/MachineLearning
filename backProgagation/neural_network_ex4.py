import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def hypothesis(theta, X):
    z = np.dot(X,theta.T)
    result = sigmoid(z)
    return result

def cost(theta, X,y):
    firstPart = -y * np.log(hypothesis(theta,X))
    secondPart = (1 - y) * np.log(1 - hypothesis(theta,X))
    m = len(X)
    resultMarix = firstPart-secondPart
    sum_ones = np.ones(np.shape(resultMarix)[1])
    sum_k = np.dot(resultMarix,sum_ones)
    return np.sum(sum_k)/m


def costWithRegularization(theta1,theta2,X,y,r_rate):
    m = len(X)
    firstPart = cost(theta2,X,y)

    theta1_without_ones = theta1[:,1:]
    theta2_without_ones = theta2[:,1:]
    theta1_power2 = np.power(theta1_without_ones,2)
    theta2_power2 = np.power(theta2_without_ones,2)

    reshapeTheta1 = np.reshape(theta1_power2,(theta1_power2.size,))
    reshapeTheta2 = np.reshape(theta2_power2,(theta2_power2.size,))

    regularization_term = r_rate/(2 * m) * (np.sum(reshapeTheta1) + np.sum(reshapeTheta2))
    return firstPart + regularization_term

def computeGradient(theta1,theta2,X,y):
    m = np.shape(X)[1]
    x_with_ones = np.insert(X,0,1,axis =1)
    l2_input = hypothesis(theta1,x_with_ones)
    l2_input_with_ones = np.insert(l2_input,0,1,axis =1)
    output = hypothesis(theta2,l2_input_with_ones)
    delta_l3 = output - y
    a = theta2.T @ delta_l3.T
    z_l2 =  l2_input_with_ones @ theta2.T
    sg_l2 = sigmoidGradient(z_l2)
    delta_l2 = a @ sg_l2
    delta_l2_without_bis = delta_l2[1:,:]

    accummulated_delta_l2 = delta_l2_without_bis + (delta_l3.T @ l2_input).T

    return accummulated_delta_l2/m







