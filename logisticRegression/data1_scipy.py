import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logisticRegression import LogisticRegression
import scipy.optimize as opt
dataPath = "..\logisticRegression\ex2data1.txt"
data = pd.read_csv(dataPath, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])



exam12 = data[['Exam 1','Exam 2']]

# dataScaled = (exam12-exam12.mean())/exam12.std()

admitted = data['Admitted']
exam12.insert(0,'Ones',1)
X = exam12
Y = admitted
m,n = np.shape(exam12)
thetas = np.zeros(n)


# cost = LogisticGradientDescenter.costFunction(exam12,admitted,thetas)
#
# gradient = LogisticGradientDescenter.gradient(exam12,admitted,thetas)

# result = opt.fmin_tnc(func=LogisticGradientDescenter.costFunction, x0=thetas, fprime=LogisticGradientDescenter.gradient, args=(X, Y))
result1 = opt.minimize(fun=LogisticRegression.costFunction, x0=thetas, args=(X, Y), method='Newton-CG', jac=LogisticRegression.gradient)

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

thetas = result1.x

print(result1)

xmax = np.max(data['Exam 1'])
x = np.arange(0,100)
y = -(thetas[0]+thetas[1]*x)/thetas[2]

fig,ax =plt.subplots(figsize=(12,8))

ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x,y)
plt.grid(True)
plt.show()