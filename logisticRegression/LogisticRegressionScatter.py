import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from numpy import genfromtxt

dataPath = "F:\PyWorkSpace\machine learning\logisticRegression\ex2data1.txt"
data = pd.read_csv(dataPath, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]



fig,ax =plt.subplots(figsize=(12,8))

ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
plt.grid(True)
plt.show()