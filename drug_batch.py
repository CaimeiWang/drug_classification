#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


pca = PCA(n_components=33)  # 加载PCA算法，设置降维后主成分数目为33
pca_data = pca.fit_transform(data)  # 对样本进行降维

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
test=[]
test_label=[]
for i in range(pca_data.shape[0]): #按9：1划分数据集
    if i<(9*pca_data.shape[0]//10):
        train.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            train_label.append([1,0,0])
        elif pca_data[i][0]==1:
            train_label.append([0,1,0])
        elif pca_data[i][0]==2:
            train_label.append([0,0,1])
    else:
        test.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            test_label.append([1,0,0])
        elif pca_data[i][0]==1:
            test_label.append([0,1,0])
        elif pca_data[i][0]==2:
            test_label.append([0,0,1])

##构建网络
innum=33
hidnum1=35
hidnum2=40
outnum=3
#初始化参数
w1=np.random.randint(-1,1,(innum, hidnum1)).astype(np.float64)  #33x35
b1=np.random.randint(-1,1,(1,hidnum1)).astype(np.float64)       #1x35
w2=np.random.randint(-1,1,(hidnum1, hidnum2)).astype(np.float64)  #35x40
b2=np.random.randint(-1,1,(1,hidnum2)).astype(np.float64)  #1x40
w3=np.random.randint(-1,1,(hidnum2,outnum)).astype(np.float64)  #40x3
b3=np.random.randint(-1,1,(1,outnum)).astype(np.float64)  #1x3
#
loopNumber=1#epoch次数
xite=0.1
batch_size=5 #批处理个数
accuracy=[]
E=[]
def softmax(z):  #softmax函数用于多分类
    z_exp=np.exp(z)
    for i in range(len(z)):
        z_exp[i]=z_exp[i]/np.sum(z_exp)
    return z_exp
def cross_entropy(X,y):   #交叉熵代价函数
    loss=np.multiply(-y,np.log(X))-np.multiply((1-y),np.log(1-X))
    return loss
dw1=np.zeros((innum,hidnum1),dtype=np.float64) #权重导数
db1=np.zeros((1,hidnum1),dtype=np.float64)#偏差导数
dw2=np.zeros((hidnum1,hidnum2),dtype=np.float64) #权重导数
db2=np.zeros((1,hidnum2),dtype=np.float64)#偏差导数
dw3=np.zeros((hidnum2,outnum),dtype=np.float64) #权重导数
db3=np.zeros((1,outnum),dtype=np.float64)#偏差导数
for j in range(loopNumber):
    e=0
    for i in range(int(np.ceil(len(train)/batch_size))):
        input_data=np.mat(train[i*batch_size:(i+1)*batch_size]).astype(np.float64)  #1x33
        pre_output=np.mat(train[i*batch_size:(i+1)*batch_size]).astype(np.float64) #1x3
        h=(np.dot(input_data,w1)+b1).astype(np.float64)  #1x35
        h_out=(1/(1+np.exp(-h))).astype(np.float64) #sigmoid激活函数
        h1=(np.dot(h_out,w2)+b2).astype(np.float64)  #1x40
        h1_out=(1/(1+np.exp(-h1))).astype(np.float64) #sigmoid激活函数
        y=(np.dot(h1_out,w3)+b3).astype(np.float64)  #1x3
        y_out1=(1/(1+np.exp(-y))).astype(np.float64)
        y_out=softmax(y_out1)#输出层输出
        #计算误差并储存

        e=cross_entropy(y_out,pre_output) #1x3
        e+=e

        E.append(np.sum(np.abs(e)))#储存损失值画损失函数
        #计算权值变化率
        db3=e
        dw3=np.multiply(e,y_out)
        w3=w3+xite*dw3
        b3=b3+xite*db3
        FI=np.multiply(h1_out,(1-h1_out)) #1x12
        e1=np.multiply(np.dot(w3,e.T).T,FI) #1*12
        db2=e1
        dw2=np.multiply(e1,h1_out)
        w2=w2+xite*dw2
        b2=b2+xite*db2
        FJ = np.multiply(h_out, (1-h_out))
        e2=np.multiply(np.dot(w2,e1.T).T,FJ)
        db1=e2
        dw1=np.multiply(e2,h_out)
        #更新权重和偏差
        w1=w1+xite*dw1
        b1=b1+xite*db1

        #储存准确率，以备画准确率曲线
        result = []
        for i in range(len(test)):
            input_data = np.mat(test[i]).astype(np.float64)
            pre_output=np.mat(test_label[i]).astype(np.float64)
            h=np.dot(input_data,w1)+b1
            h_out=1/(1+np.exp(-h))
            h1=np.dot(h_out,w2)+b2
            h1_out=1/(1+np.exp(-h1))
            y=np.dot(h1_out,w3)+b3
            y_out1=1/(1+np.exp(-y))
            y_out=softmax(y_out1)
            print(y_out)
            if np.argmax(y_out)==np.argmax(pre_output):
                result.append(1)
            else:
                result.append(0)
        pre_result=np.sum(result)/len(result)
        accuracy.append(pre_result)
print('E:',E)
print('accuracy:',accuracy)
# count=[]
# for i in range(len(accuracy)):
#     count.append(i)
# plt.subplot(212)
# plt.plot(count,accuracy,'r-')
# plt.xlabel('iterations')
# plt.ylabel('accuracy')
# x=[]
# for i in range(len(E)):
#     x.append(i)
# plt.subplot(211)
# plt.plot(x,E,'g-')
# plt.xlabel('iterations')
# plt.ylabel('loss')
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.show()
# #print(max(accuracy))