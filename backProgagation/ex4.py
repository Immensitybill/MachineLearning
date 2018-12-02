
from scipy.io import loadmat
import pandas as pd
import numpy as np

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



def run():
    theta1, theta2 = loadTheta()
    theta = neural_network_ex4.serialize(theta1, theta2)
    X_raw,y_raw=loadData()
    X_added_bis = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
    y_processed = dataProcess(y_raw)
    cost = neural_network_ex4.cost(theta,X_added_bis,y_processed)
    print(cost)

    # X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
    # # y = expand_y(y_raw)
    # y_processed = dataProcess(y_raw)
    # theta =  neural_network_ex4.serialize(theta1,theta2)
    # g = neural_network_ex4.computeGradient(theta,X_raw,y_processed)
    # neural_network_ex4.gradient_checking(theta,X,y_processed,0.0001)
    # print(np.shape(theta))
    # nn_back_propagation.gradient_checking(theta,X,y_processed,0.0001)

run()