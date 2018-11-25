import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from logisticRegression import LogisticGradientDescenter, RegularizedLogisticRegression

data = pd.DataFrame()
test12 = pd.DataFrame()

def readData():
    dataPath = "..\logisticRegression\ex2data2.txt"
    global data
    global test12
    data = pd.read_csv(dataPath, header=None, names=['test1', 'test2', 'Admitted'])
    test12 = data[['test1','test2']]


def mapFeature(X,degree):
    degree = degree+1
    x1 = X['test1']
    x2 = X['test2']
    for i in range(0, degree):
        for j in range (0 , degree-i):
            X ['F'+ str(i)+str(j)] = np.power(x1,i)*np.power(x2,j)
    X.drop('test1',axis=1,inplace=True)
    X.drop('test2',axis=1,inplace=True)
    return X


def run():
    readData()
    degree = 6

    X = mapFeature(test12,degree)
    Y = data['Admitted']
    m,n = np.shape(X)
    thetas = np.zeros(n)
    lamda = 0.2

    # cost = RegularizedLogisticRegression.costFunction(thetas,X, Y, 1)
    g0 = RegularizedLogisticRegression.gradient(thetas,X,Y,1)

    result = opt.fmin_tnc(func=RegularizedLogisticRegression.costFunction, x0=thetas, fprime=RegularizedLogisticRegression.gradient, args=(X, Y,lamda))
    print(result[0])

    x,y = findDescitionBoundary(result[0],degree)
    draw_boundary(x,y)



def findDescitionBoundary(theta,degree):
    t1 = np.linspace(-1,1.5,1000)
    t2 = np.linspace(-1,1.5,1000)
    cordinates = [(x,y) for x in t1 for y in t2]

    cordFrame = pd.DataFrame(cordinates)

    cordFrame.columns = ['test1','test2']

    mappedCordFrame = mapFeature(cordFrame,degree)

    inner_product = mappedCordFrame.as_matrix() @ theta

    decision = mappedCordFrame[np.abs(inner_product) < 2 * 10**-3]

    return decision.F10, decision.F01


def draw_boundary(X,Y):
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    fig,ax =plt.subplots(figsize=(12,8))

    ax.scatter(X,Y,c='y')
    ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Not Admitted')
    plt.grid(True)
    plt.show()

run()

# hy = LogisticGradientDescenter.hypothesis(result[0],X)
#
# result = []
# for i in hy:
#     if i >= 0.5:
#         result.append(1)
#     else:
#         result.append(0)
#
# sum = 0
# for j in range(0,len(result)):
#     if result[j] == Y[j]:
#         sum = sum+1
#
# print ("correct ratio: ",sum/len(result))



