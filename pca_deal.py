#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', 100000)
#数据处理
with open('data.csv','r',encoding='ISO-8859-1') as f: #读取文件
    data= csv.reader(f)
    data=list(data)
    data[0][0] = 1433  #修正异常数据
    data=np.array(data).astype(np.float64)  #将样本存放到 一个大矩阵中
    data=data.T

data=scale(data,axis=1)   #标准化
pca = PCA(n_components=0.90)  #加载PCA算法，设置降维后主成分数目为30
pca_data=pca.fit_transform(data)  #对样本进行降维
#pca.fit(data)
print (pca.explained_variance_ratio_)
print(pca_data.shape)
a1=[];a2=[];a3=[]
b1=[];b2=[];b3=[]
c1=[];c2=[];c3=[]
for i in range(1000):
    a1.append(pca_data[i][0])
    a2.append(pca_data[i][1])
    a3.append(pca_data[i][2])
for i in range(1000,2000):
    b1.append(pca_data[i][0])
    b2.append(pca_data[i][1])
    b3.append(pca_data[i][2])
for i in range(2000,3000):
    c1.append(pca_data[i][0])
    c2.append(pca_data[i][1])
    c3.append(pca_data[i][2])
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(a1,a2,a3, c='r', marker='x')
plt.scatter(b1,b2,b3, c='b', marker='D')
plt.scatter(c1,c2,c3, c='g', marker='.')
plt.show()