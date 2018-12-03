
from scipy.io import loadmat
import pandas as pd
import numpy as np
import scipy.optimize as opt

import scipy.io as sio
import csv
from sklearn.metrics import classification_report

# from backProgagation import neural_network_ex4, nn_back_propagation
from backProgagation import neural_network_ex4
from neuralNetwork import neural_network

def loadTheta():
    data = loadmat('../backProgagation/ex4weights.mat')
    theta1= data['Theta1']
    theta2= data['Theta2']
    return theta1,theta2

def loadData():
    data = loadmat('../backProgagation/ex4data1.mat')
    X= data['X']
    Y= data['y']
    return X,Y

def dataProcess(y):
    labels = np.unique(y)
    y_matrix = pd.DataFrame()
    for label in labels:
        y_matrix[label-1] = [1 if label == y_i else 0 for y_i in y]
    return y_matrix

def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(X, y):
    """regularized version
    the architecture is hard coded here... won't generalize
    """
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=neural_network_ex4.costWithRegularization,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=neural_network_ex4.computeGradientRegularization,
                       options={'maxiter': 400})
    return res
def show_accuracy(theta, X, y):
    _, _, _, _, h = neural_network_ex4.feedForward(theta, X)

    y_pred = np.argmax(h, axis=1) + 1

    print(classification_report(y, y_pred))

def run():
    theta1, theta2 = loadTheta()
    theta = neural_network_ex4.serialize(theta1, theta2)
    X_raw,y_raw=loadData()
    X_added_bis = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
    y_processed = dataProcess(y_raw)
    res = nn_training(X_added_bis,y_processed)
    print(res)
    final_theta = res.x

    show_accuracy(final_theta,X_added_bis,y_raw)

    # cost = neural_network_ex4.costWithRegularization(theta,X_added_bis,y_processed,r_rate=1)
    # print(cost)

    # neural_network_ex4.gradient_checking(theta,X_added_bis,y_processed,0.0001,True)

    # neural_network_ex4.computeGradientRegularization(theta,X_added_bis,y_processed)


    # X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
    # # y = expand_y(y_raw)
    # y_processed = dataProcess(y_raw)
    # theta =  neural_network_ex4.serialize(theta1,theta2)
    # g = neural_network_ex4.computeGradient(theta,X_raw,y_processed)
    # neural_network_ex4.gradient_checking(theta,X,y_processed,0.0001)
    # print(np.shape(theta))
    # nn_back_propagation.gradient_checking(theta,X,y_processed,0.0001)

run()