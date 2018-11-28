import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

aa =np.array([[11,33,44],[22,33,44],[55,66,77]])

bb = np.array([[22,33,55],[33,33,44],[66,66,66]])

# aa.columns = ["test1",'test2','test3']

# bb = np.ones(aa.shape[1])
#
# y = aa @ bb
# y1 = np.dot(aa,bb)
print(aa== bb)

cc = [1,2,3,4,5,6,7,8,9,10]


for i in range(1,11):
    result = [1 if i == c else 0 for c in cc]
    print(result)





