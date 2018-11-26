import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

a = np.array([1,4,7]).T

aa = a.T

b = np.array([[2,3],[5,6],[8,9]])

c = np.array([2,3,4])

result = np.dot(a.T,c)

print(aa)

