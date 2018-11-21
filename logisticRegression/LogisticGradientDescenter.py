import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataPath = "F:\PyWorkSpace\machine learning\logisticRegression\ex2data1.txt"
data = pd.read_csv(dataPath, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

exam12 = data[['Exam 1','Exam 2']]

def sigmoid(z):
    return 1/(1+np.exp(-z))



def hypothesis (X,thetas):
    z = np.dot(X,thetas)
    return sigmoid(z)

m,n = np.shape(exam12)
thetas = np.zeros(n)

def costFunction(X,Y,thetas):

   np.sum (-Y*np.log(hypothesis(X,thetas)-(1-Y))*np.log(1-hypothesis(X,thetas)))/X.len()

hypothesis(exam12,thetas)

# x = np.arange(-10,10)
# y = sigmoid(x)
#
# plt.plot(x,y),
#
# plt.show()

