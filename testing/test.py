import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv('F:\PyWorkSpace\machine learning\linearRegression\ex1data1.txt', names=['population', 'profit'])
print(df.head())

tf.train.GradientDescentOptimizer