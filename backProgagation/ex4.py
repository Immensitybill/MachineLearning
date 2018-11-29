
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
    y_matrix = pd.DataFrame()
    for label in labels:
        y_matrix[label-1] = [1 if label == y_i else 0 for y_i in y]
    return y_matrix


def run():
    theta1, theta2 = loadTheta()
    X,y=loadData()
    y_processed = dataProcess(y)
    g_l2 =  neural_network_ex4.computeGradient(theta1,theta2,X,y_processed)
    print()

run()