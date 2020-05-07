#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import *
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import learning_curve
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

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
pca = PCA(n_components=30)  #加载PCA算法，设置降维后主成分数目为30
pca_data=pca.fit_transform(data)  #对样本进行降维



#制作特征样本和分类标签
pca_data= pca_data.tolist()
for i in range(3000):     #先加入0，1，2标签，制作带标签数据集
    if i<1000:
        pca_data[i].insert(0,0)
    elif i<2000:
        pca_data[i].insert(0,1)
    elif i<3000:
        pca_data[i].insert(0,2)
shuffle_num= np.random.permutation(len(pca_data)) #打乱数据集顺序
pca_data=np.array(pca_data)[shuffle_num]

#将特征和标签分开，并把标签制成独热型
train=[]
train_label=[]
val=[]
val_label=[]
test=[]
test_label=[]
test_label1=[]
for i in range(pca_data.shape[0]): #按8:1:1划分数据集
    if i<(8*pca_data.shape[0]//10):
        train.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            train_label.append(0)
        elif pca_data[i][0]==1:
            train_label.append(1)
        elif pca_data[i][0]==2:
            train_label.append(2)
    elif i in range(8*pca_data.shape[0]//10,9*pca_data.shape[0]//10):
        val.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            val_label.append([1,0,0])
        elif pca_data[i][0]==1:
            val_label.append([0,1,0])
        elif pca_data[i][0]==2:
            val_label.append([0,0,1])
    else:
        test.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            test_label.append([1,0,0])
            test_label1.append(0)
        elif pca_data[i][0]==1:
            test_label.append([0,1,0])
            test_label1.append(1)
        elif pca_data[i][0]==2:
            test_label.append([0,0,1])
            test_label1.append(2)
train=np.array(train)
train_label=np.array(train_label)
val=np.array(val)
val_label=np.array(val_label)
test=np.array(test)
test_label=np.array(test_label)
test_label1=np.array(test_label1)
#Bayes 调节函数
# from sklearn.naive_bayes import MultinomialNB #多项式型
# mnb=MultinomialNB(alpha=1)
# Bayes_model=mnb.fit(train,train_label)
# from sklearn.naive_bayes import GaussianNB #高斯分布型
# gnb=GaussianNB()
# Bayes_model1=gnb.fit(train,train_label)
# y_pred1=Bayes_model1.predict(test)
# y_true1=test_label1
# a1=[]
# for i in range(len(y_pred1)):
#     if y_pred1[i]==y_true1[i]:
#         a1.append(1)
#     else:
#         a1.append(0)
# accuracy1=sum(a1)/len(a1)

from sklearn.naive_bayes import BernoulliNB #伯努利型
bnb=BernoulliNB()
Bayes_model=bnb.fit(train,train_label)
y_pred=Bayes_model.predict(test)
y_true=test_label1
# a=[]
# for i in range(len(y_pred)):
#     if y_pred[i]==y_true[i]:
#         a.append(1)
#     else:
#         a.append(0)
# accuracy2=sum(a)/len(a)
# # 显示高度
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2, 1.03*height, '%s' %float('%.2f'%height))
# name_list = ['高斯分布型', '伯努利型',]
# num_list = [accuracy1,accuracy2]
# rect=plt.bar([0.1,0.2],num_list,color='rgb',width =0.02,tick_label=name_list,yerr=0.000001,align="center")
# autolabel(rect)
# plt.xticks=np.arange(0,0.3,0.1)
# plt.yticks(np.arange(0,1,0.1))
# plt.xlabel('核函数')
# plt.ylabel('准确率')
# plt.show()
y_pred1=[]
for i in range(len(y_pred)):
    if y_pred[i]==0:
        y_pred1.append([1,0,0])
    if y_pred[i] ==1:
        y_pred1.append([0,1, 0])
    if y_pred[i] ==2:
        y_pred1.append([0, 0,1])
np.save('y_true_bays.npy',test_label)
np.save('y_pred_bays.npy',y_pred1)
def confusion_metrix(y,y_p):
    Confusion_matrix=confusion_matrix(y,y_p) #y代表真实值，y_p 代表预测值
    plt.matshow(Confusion_matrix)
    plt.title("混淆矩阵")
    plt.colorbar()
    plt.ylabel("实际类型")
    plt.xlabel("预测类型")
    plt.show()

if __name__ == '__main__':
    confusion_metrix(y_true,y_pred)