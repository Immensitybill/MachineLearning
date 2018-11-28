
from scipy.io import loadmat
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import classification_report

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
    for label in labels:
        y_matrix = [1 if label == y_i else 0 for y_i in y]
    return y_matrix


def run():
    theta1, theta2 = loadTheta()
    X,y=loadData()

    y_processed = pd.DataFrame(dataProcess(y))




    input = np.insert(X,0,1,axis=1)
    L1 = neural_network_ex4.hypothesis(theta1.T,input)
    L1I = np.insert(L1,0,1,axis=1)
    output = neural_network_ex4.hypothesis(theta2.T,L1I)
    cost = neural_network_ex4.cost(X,y,output)
    print(cost)

run()