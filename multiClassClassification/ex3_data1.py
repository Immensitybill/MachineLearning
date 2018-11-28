from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import pandas as pd
from scipy.optimize import minimize

from multiClassClassification import LogisticRegression_ex3_data1


def loadData():
    data = loadmat('../multiClassClassification/ex3data1.mat')
    X= data['X']
    Y= data['y']
    return X,Y


def run():
    X, Y = loadData()
    X1 = np.insert(X,0,1,axis=1)
    labels = np.unique(Y)
    m, n = np.shape(X1)
    lamda = 1
    thetas = pd.DataFrame()
    for i in labels:
        theta = np.zeros(n)
        y_i = np.array([1 if i == label else 0 for label in Y])
        result = opt.fmin_tnc(func=LogisticRegression_ex3_data1.costFunction, x0=theta,fprime=LogisticRegression_ex3_data1.regularized_gradient, args=(X1, y_i, lamda))
        # fmin = minimize(fun=LogisticRegression_ex3_data1.cost, x0=theta, args=(X1, y_i, lamda), method='TNC', jac=LogisticRegression_ex3_data1.regularized_gradient)
        thetas[i-1]=result[0]
        print('thetas: ',i,': ',result[0])


    y_pred = predict_all(X, thetas)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, Y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[1]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = LogisticRegression_ex3_data1.sigmoid(X * all_theta)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


run()

