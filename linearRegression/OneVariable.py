from linearRegression.GradientDescenter import  GradientDescenter
from linearRegression import DataGetter
import numpy as np
from matplotlib import pyplot as plt

dataPath = "F:\PyWorkSpace\machine learning\linearRegression\ex1data1.txt"
trainingDatas, labels = DataGetter.getData(dataPath)
m, n = np.shape(trainingDatas)

theta = GradientDescenter.batchGradientDescent(trainingDatas,labels,0.01,1000)


population = trainingDatas[:,:-1]
x = np.linspace(population.min(),population.max(),100)
y = theta[0]*x + theta[1]
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, y,'r',label='Prediction')
ax.scatter(trainingDatas[:,0],labels,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()