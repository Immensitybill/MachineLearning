from scipy.io import loadmat
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import classification_report

from neuralNetwork import neural_network


def loadTheta():
    data = loadmat('../neuralNetwork/ex3weights.mat')
    theta1= data['Theta1']
    theta2= data['Theta2']
    return theta1,theta2

def loadData():
    data = loadmat('../multiClassClassification/ex3data1.mat')
    X= data['X']
    Y= data['y']
    return X,Y

def run():
    theta1, theta2 = loadTheta()
    X,Y=loadData()
    m,n = X.shape
    X1=np.insert(X,0,1,axis=1)

    test = X1[:,:]
    # export = pd.DataFrame(test)
    # export.to_csv('export.csv',index = False)

    # print('test',test)
    L1 = neural_network.hypothesis(theta1.T,test)

    L1I = np.insert(L1,0,1,axis=1)

    output = neural_network.hypothesis(theta2.T,L1I)

    predict = np.argmax(output, axis=1) + 1
    print(classification_report(Y,predict))

run()