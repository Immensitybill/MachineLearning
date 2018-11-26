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

# 下面这个是错的，因为下面的实现实际上把同一个hypothesis拆成了2个，等于说在考虑gradient1的时候改变了原hypothesis,而没有考虑theta0(截距项)的因素。
# 我们在做regularzed的regression的时候，只是在j=0的时候不加上reguler项，其实不管是j=0和j>=1的时候，假设函数都没有变化，
# 都需要考虑theta0的因素,但是在下面gradient1的时候X1已经去掉了'F00'因此假设函数并不跟原来一样，导致错误的发生。
# def gradient (thetas,X,Y,lamda):
#     X0 = X['F00']
#     theta0 = thetas[0]
#     a0 = hypothesis(theta0, X0) - Y
#     gradient0 = np.dot(X0.T, a0) / len(X0)
#
#
#     X1 = X.drop(['F00'],axis=1)
#     theta1 = np.delete(thetas,0,axis=0)
#     a1 = hypothesis(theta1,X1)-Y
#     gradient1 = (np.dot(X1.T,a1)/len(X1))+(lamda/len(X1)*theta1)
#
#     result = np.insert(gradient1,0,values=gradient0,axis=0)
#     print(result)
#     return result

def regularized_gradient(theta, X, y, l):
#     '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    g = gradient(theta, X, y)
    result = g + regularized_term
    print(result)
    return result