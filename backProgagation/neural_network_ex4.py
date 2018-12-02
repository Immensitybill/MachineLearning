import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def hypothesis(theta, X):
    z = np.dot(X,theta.T)
    result = sigmoid(z)
    return result

def cost(theta, X_with_bis, y_processed):
    _, _, _, _, h = feedForward(theta, X_with_bis)
    firstPart = -y_processed * np.log(h)
    secondPart = (1 - y_processed) * np.log(1 - h)
    m = len(X_with_bis)
    resultMarix = firstPart-secondPart
    sum_ones = np.ones(np.shape(resultMarix)[1])
    sum_k = np.dot(resultMarix,sum_ones)
    return np.sum(sum_k)/m

def feedForward(theta, X):
    theta1, theta2 = deserialize(theta)
    a1_with_ones = X
    z2 = np.dot(a1_with_ones, theta1.T)
    a2 = hypothesis(theta1, a1_with_ones)
    a2_with_ones = np.insert(a2, 0, 1, axis=1)
    z3 = np.dot(a2_with_ones, theta2.T)
    a3 = hypothesis(theta2, a2_with_ones)
    return a1_with_ones, z2, a2_with_ones, z3, a3

def costWithRegularization(theta,X,y,r_rate):
    theta1,theta2 = deserialize(theta)
    m = len(X)
    firstPart = cost(theta,X,y)

    theta1_without_ones = theta1[:,1:]
    theta2_without_ones = theta2[:,1:]
    theta1_power2 = np.power(theta1_without_ones,2)
    theta2_power2 = np.power(theta2_without_ones,2)

    reshapeTheta1 = np.reshape(theta1_power2,(theta1_power2.size,))
    reshapeTheta2 = np.reshape(theta2_power2,(theta2_power2.size,))

    regularization_term = r_rate/(2 * m) * (np.sum(reshapeTheta1) + np.sum(reshapeTheta2))
    return firstPart + regularization_term

def computeGradient(theta,X,y):
    theta1, theta2 = deserialize(theta)
    m = np.shape(X)[0]
    a1_with_ones = np.insert(X,0,1,axis =1)
    a2 = hypothesis(theta1,a1_with_ones)
    a2_with_ones = np.insert(a2,0,1,axis =1)
    a3 = hypothesis(theta2,a2_with_ones)
    d3 = a3 - y
    delta2 = a2_with_ones.T @ d3

    z2 = np.dot(a1_with_ones, theta1.T)
    z2 = np.insert(z2,0,1,axis=1)
    sigGrad = sigmoidGradient(z2)
    b = theta2.T @ d3.T
    d2 = np.multiply(b.T , sigGrad)
    delta1 = a1_with_ones.T @ d2[:,1:]

    return serialize(delta1.T/m, delta2.T / m)

def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))

def gradient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        """calculate a partial gradient with respect to 1 theta"""
        if regularized:
            return (costWithRegularization(plus, X, y) - costWithRegularization(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    # calculate numerical gradient with respect to all theta
    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                                    for i in range(len(theta))])

    # analytical grad will depend on if you want it to be regularized or not
    analytic_grad = computeGradient(theta, X, y)

    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # the diff below should be less than 1e-9
    # this is how original matlab code do gradient checking
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

def expand_array(arr):
    """replicate array into matrix
    [1, 2, 3]

    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    """
    # turn matrix back to ndarray
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

def deserialize(seq):
#     """into ndarray of (25, 401), (10, 26)"""
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)