import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

a = np.identity(4) *2
b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,5,6]])

product = np.dot(a,b)
product1 = np.dot(b,a)

print(product)

# cc = [1,2,3,4,5,6,7,8,9,10,1,2,3]
#
# m= len(np.unique(cc))
# n= len(cc)
#
# result = pd.DataFrame()
#
# for i in range(1,11):
#     result[i-1] = [1 if i == c else 0 for c in cc]

# sum = np.sum(np.reshape(aa,(aa.size,)))

# print(aa)
# print(aa.size)



