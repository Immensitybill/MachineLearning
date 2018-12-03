import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

a = np.array([[0,0,0],[0,0,0],[0,0.0001,0]])
b = np.identity(3) * 2

print ('a*b',np.dot(a,b))
print ('b*a',np.dot(b,a))

print('a+b',a+b)

print(np.linalg.norm(a))

print()



def expand_array(arr):
    """replicate array into matrix
    [1, 2, 3]

    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    """
    # turn matrix back to ndarray
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

theta = np.array([1,2,3])
theta_matrix = expand_array(theta)  # expand to (10285, 10285)
epsilon_matrix = np.identity(len(theta)) * 0.1

plus_matrix = theta_matrix + epsilon_matrix
minus_matrix = theta_matrix - epsilon_matrix


print()

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



