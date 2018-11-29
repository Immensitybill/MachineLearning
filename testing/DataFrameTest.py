import pandas as pd
import numpy as np

aa = pd.DataFrame([[11,22,33],[44,55,66],[77,88,99]])

bb = np.array([[111,222,333],[444,555,666],[777,888,999]])

#获取第一列（colums）:11,44,55
cc = aa[0]

#获取第一行（DataFrame格式）：11,22,33
cc = aa[0:1]

# 获取第一行（Series）:11,22,33
cc = aa.iloc[[0],[0]]

# 根据index获取第一行（Series）:11,22,33
cc = aa.iloc[0,:]

#根据index获取第一行和第二行（DataFrame格式）
cc = aa.iloc[[0,1],:]

#根据名称获取第一行,因为正好这里只有数字索引，所以可以
cc = aa.loc[0]

#用ix的时候既可以用索引（数字），也可以用名字
aa.columns = ['a','b','c']
cc = aa.ix[:,0]
cc = aa.ix[:,'a']

#修改行名称
aa.index = ['aa','bb','cc']

print(aa)