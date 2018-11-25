import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

a = pd.DataFrame([[1,2],[2,3],[3,4]])

b = a.drop([0],axis=1)

print(b)