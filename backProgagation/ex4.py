
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
                       options={'maxiter': 800})
    return res
def show_accuracy(theta, X, y):
    _, _, _, _, h = neural_network_ex4.feedForward(theta, X)

    y_pred = np.argmax(h, axis=1) + 1

    print(classification_report(y, y_pred))

def getTrainingData(X_raw,y_raw):
    X_training = X_raw[0:250,:]
    y_training = y_raw[0:250,:]
    a=500
    for i in range(1,10):
        X_training = np.concatenate((X_training,X_raw[a:a+250,:]))
        y_training = np.concatenate((y_training,y_raw[a:a+250,:]))
        a=a+500
    return X_training,y_training
    # X_training = pd.DataFrame(dtype="float64")
    # y_training = pd.DataFrame()
    # a = 0
    # for i in range(1,11):
    #     # X_training = pd.concat([X_training,pd.DataFrame(X_raw[a:a+250])])
    #     X_training = X_training.append(pd.DataFrame(X_raw[a:a+250]),ignore_index=True)
    #     y_training = y_training.append(pd.DataFrame(y_raw[a:a+250]), ignore_index=True)
    #     a = a + 500
    # return X_training,y_training
        # x10 = X_raw[0:250]
        # x1 = X_raw[500:750]
        # x2 = X_raw[1000:1250]

def getPredictionData(X_raw,y_raw):
    X_predict = X_raw[250:500, :]
    y_predict = y_raw[250:500, :]
    a = 750
    for i in range(1, 10):
        X_predict = np.concatenate((X_predict, X_raw[a:a + 250, :]))
        y_predict = np.concatenate((y_predict, y_raw[a:a + 250, :]))
        a = a + 500
    return X_predict, y_predict

def run():
    theta1, theta2 = loadTheta()
    theta = neural_network_ex4.serialize(theta1, theta2)
    X_raw,y_raw=loadData()
    X_training_raw,y_training_raw = getTrainingData(X_raw,y_raw)
    X_predict_raw,y_predict_raw =getPredictionData(X_raw, y_raw)
    X_added_bis = np.insert(X_training_raw, 0, np.ones(X_training_raw.shape[0]), axis=1)
    X_predict_added_bis = np.insert(X_predict_raw, 0, np.ones(X_predict_raw.shape[0]), axis=1)

    y_processed = dataProcess(y_training_raw)
    res = nn_training(X_added_bis,y_processed)
    print(res)
    final_theta = res.x

    show_accuracy(final_theta,X_predict_added_bis,y_predict_raw)

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