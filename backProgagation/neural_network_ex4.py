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

def feedForward(theta, X_with_bis):
    theta1, theta2 = deserialize(theta)
    a1_with_ones = X_with_bis
    z2 = np.dot(a1_with_ones, theta1.T)
    a2 = hypothesis(theta1, a1_with_ones)
    a2_with_ones = np.insert(a2, 0, 1, axis=1)
    z3 = np.dot(a2_with_ones, theta2.T)
    a3 = hypothesis(theta2, a2_with_ones)
    return a1_with_ones, z2, a2_with_ones, z3, a3

def costWithRegularization(theta, X_with_bis, y_processed, r_rate=1):
    theta1,theta2 = deserialize(theta)
    m = len(X_with_bis)
    firstPart = cost(theta, X_with_bis, y_processed)

    theta1_without_ones = theta1[:,1:]
    theta2_without_ones = theta2[:,1:]
    theta1_power2 = np.power(theta1_without_ones,2)
    theta2_power2 = np.power(theta2_without_ones,2)

    reshapeTheta1 = np.reshape(theta1_power2,(theta1_power2.size,))
    reshapeTheta2 = np.reshape(theta2_power2,(theta2_power2.size,))

    regularization_term = r_rate/(2 * m) * (np.sum(reshapeTheta1) + np.sum(reshapeTheta2))
    return firstPart + regularization_term

def computeGradient(theta, X_with_bis, y_processed):
    theta1, theta2 = deserialize(theta)#theta1 (25,401) theta2(10,26)
    m = np.shape(X_with_bis)[0]
    # a1_with_ones = np.insert(X_with_bis, 0, 1, axis =1)
    a2 = hypothesis(theta1,X_with_bis) #(5000,25)
    a2_with_ones = np.insert(a2,0,1,axis =1) #(5000,26)
    a3 = hypothesis(theta2,a2_with_ones) #(5000,10)
    d3 = a3 - y_processed #(5000 10)
    delta2 = a2_with_ones.T @ d3 #(26 10)

    z2 = np.dot(X_with_bis, theta1.T)
    z2 = np.insert(z2,0,1,axis=1) #(5000,26)
    sigGrad = sigmoidGradient(z2) #(5000,26)
    b = theta2.T @ d3.T #(26 5000)
    d2 = np.multiply(b.T , sigGrad) #(5000,26)
    delta1 = X_with_bis.T @ d2[:,1:] #(401,25)

    return serialize(delta1.T/m, delta2.T / m)

def computeGradientRegularization(theta, X_with_bis, y_processed,r_rate=1):
    m = np.shape(X_with_bis)[0]
    theta1, theta2 = deserialize(theta)
    theta1_regularization_term = r_rate/m * theta1
    theta2_regularization_term = r_rate / m * theta2
    delta = computeGradient(theta,X_with_bis,y_processed)
    delta1,delta2 = deserialize(delta)
    theta1_regularization_term = theta1_regularization_term[:,1:]
    theta2_regularization_term = theta2_regularization_term[:,1:]

    theta1_regularization_term = np.insert(theta1_regularization_term,0,0,axis=1)
    theta2_regularization_term = np.insert(theta2_regularization_term, 0, 0, axis=1)

    g1 = delta1 + theta1_regularization_term
    g2 = delta2 + theta2_regularization_term

    return serialize(g1,g2)




def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))

def gradient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        """calculate a partial gradient with respect to 1 theta"""
        if regularized:
            return (costWithRegularization(plus, X, y) - costWithRegularization(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    #这里的目的就是可以把每个theta都可以改变一下，都带入到a_numeric_grad算一下
    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix # theta_matrix对角线上每个元素加epsilon
    minus_matrix = theta_matrix - epsilon_matrix # theta_matrix对角线上每个元素减epsilon

    # calculate numerical gradient with respect to all theta
    #这里相当于对每个theta都改变了一点点，然后都带入到a_numeric_grad算了一下
    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                                    for i in range(len(theta))])

    # analytical grad will depend on if you want it to be regularized or not

    analytic_grad = computeGradientRegularization(theta,X,y) if regularized else computeGradient(theta, X, y)

    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # the diff below should be less than 1e-9
    # this is how original matlab code do gradient checking
    # norm可以计算某个矩阵/向量的范数，指的是向量x到原点的距离，或者可以说是一个向量或者矩阵的大小。
    # 这里（numeric_grad - analytic_grad）的范数应该是个几乎是0的数，（numeric_grad + analytic_grad）应该大致是单独analytic_grad的两倍，
    # 相除的结果即diff应该是个几乎是0的数

    a = np.linalg.norm(numeric_grad - analytic_grad)
    print("a:",a)
    b = np.linalg.norm(numeric_grad + analytic_grad)
    print("b:",b)
    diff = a / b

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