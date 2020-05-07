#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

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
pca = PCA(n_components=8)  #加载PCA算法，设置降维后主成分数目为30
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
innum=8
hidnum1=20
hidnum2=12
outnum=3
#初始化参数
w1=np.random.randint(-1,1,(innum, hidnum1)).astype(np.float64)
b1=np.random.randint(-1,1,(1,hidnum1)).astype(np.float64)
w2=np.random.randint(-1,1,(hidnum1, hidnum2)).astype(np.float64)
b2=np.random.randint(-1,1,(1,hidnum2)).astype(np.float64)
w3=np.random.randint(-1,1,(hidnum2,outnum)).astype(np.float64)
b3=np.random.randint(-1,1,(1,outnum)).astype(np.float64)
w3_1=w3
w2_1=w2
w1_1=w1
b1_1=b1
b2_1=b2
b3_1=b3
loopNumber=1#epoch次数
xite=0.01
batch_size=5
accuracy=[]
E=[]
def sigmoid(x):  #sigmoid函数
    x=np.array(x)
    for i in range(x.shape[1]):
        if x[0][i]>=0:      #对sigmoid函数的优化，避免出现极大的数据溢出
            x[0][i]=1.0/(1+np.exp(-x[0][i]))
        else:
            x[0][i]=np.exp(x[0][i])/(1+np.exp(x[0][i]))
    return x
def relu(x):
    x=np.array(x)
    for i in range(x.shape[1]):
        if x[0][i]>0:
            x[0][i]=1.0/(1+np.exp(-x[0][i]))
        else:
            x[0][i]=np.exp(x[0][i])/(1+np.exp(x[0][i]))
    return x
def back_relu(x):
    x=np.array(x)
    for i in range(x.shape[1]):
        if x[0][i]>0:
            x[0][i]=1
        else:
            x[0][i]=0
    return x

def softmax(z):  #softmax函数用于多分类
    z_exp=np.exp(z)
    for i in range(len(z)):
        z_exp[i]=z_exp[i]/np.sum(z_exp)
    return z_exp
def cross_entropy(X,y):   #交叉熵代价函数
    loss=-(np.multiply(y,np.log(X))+np.multiply((1-y),np.log(1-X)))
    return loss
dw1=np.zeros((innum,hidnum1),dtype=np.float64) #权重导数
db1=np.zeros((1,hidnum1),dtype=np.float64)#偏差导数
dw2=np.zeros((hidnum1,hidnum2),dtype=np.float64) #权重导数
db2=np.zeros((1,hidnum2),dtype=np.float64)#偏差导数
dw3=np.zeros((hidnum2,outnum),dtype=np.float64) #权重导数
db3=np.zeros((1,outnum),dtype=np.float64)#偏差导数
e=0
for j in range(loopNumber):
    for i in range(len(train)):
        input_data=np.mat(train[i]).astype(np.float64)
        pre_output=np.mat(train_label[i]).astype(np.float64)
        h=(np.dot(input_data,w1)+b1).astype(np.float64)
        #h_out=sigmoid(h) #sigmoid激活函数
        h_out=relu(h)  #relu激活函数
        h1=(np.dot(h_out,w2)+b2).astype(np.float64)
        #h1_out=sigmoid(h1) #sigmoid激活函数
        h1_out=relu(h1)  #relu激活函数
        y=(np.dot(h1_out,w3)+b3).astype(np.float64)
        #y_out1=sigmoid(y)#sigmoid激活函数
        y_out1=relu(y)#relu激活函数
        y_out=softmax(y_out1)#输出层输出
        #计算误差并储存
        e+=cross_entropy(y_out,pre_output) #1x3 #交叉熵代价函数
        #e+=(1/(2*outnum))*np.multiply((y_out-pre_output),(y_out-pre_output)) #均方差代价函数
        #e+= pre_output - y_out
        if (i+1)%batch_size==0:
            e=e/batch_size
            E.append(np.sum(np.abs(e)))  # 储存损失值画损失函数
            #计算权值变化率
            db3=e
            dw3=np.multiply(e,y_out)
            w3=w3_1+xite*dw3
            b3=b3_1+xite*db3
            #FI=np.multiply(h1_out,(1-h1_out))#sigmoid
            FI=back_relu(h1_out) #relu
            e1=np.multiply(np.dot(w3,np.transpose(e)).T,FI)
            db2=e1
            dw2=np.multiply(e1,h1_out)
            w2=w2_1+xite*dw2
            b2=b2_1+xite*db2
            #FJ=np.multiply(h_out, (1-h_out))#sigmoid
            FJ=back_relu(h_out)  #relu
            e2=np.multiply(np.dot(w2,e1.T).T,FJ)
            db1=e2
            dw1=np.multiply(e2,h_out)
            #更新权重和偏差
            w1=w1_1+xite*dw1
            b1=b1_1+xite*db1
            w1_1=w1
            w2_1=w2
            w3_1=w3
            b1_1=b1
            b2_1=b2
            b3_1=b3
            e=0
            # 储存准确率，以备画准确率曲线
            result = []
            for i in range(len(test)):
                input_data = np.mat(test[i]).astype(np.float64)
                pre_output=np.mat(test_label[i]).astype(np.float64)
                h=np.dot(input_data,w1)+b1
                #h_out=sigmoid(h) #sigmoid
                h_out=relu(h)  #relu
                h1=np.dot(h_out,w2)+b2
                #h1_out=sigmoid(h1) #sigmoid
                h1_out=relu(h1)  #relu
                y=np.dot(h1_out,w3)+b3
                #y_out1=sigmoid(y) #sigmoid
                y_out1=relu(y) #relu
                y_out=softmax(y_out1)
                if np.argmax(y_out)==np.argmax(pre_output):
                    result.append(1)
                else:
                    result.append(0)
            pre_result=np.sum(result)/len(result)
            print(pre_result)
            accuracy.append(pre_result)
        else:
            continue

# print('E:',E)
# print('accuracy:',accuracy)
count=[]
for i in range(len(accuracy)):
    count.append(i)
plt.subplot(212)
plt.plot(count,accuracy,'r-')
plt.xlabel('iterations')
plt.ylabel('accuracy')
x=[]
for i in range(len(E)):
    x.append(i)
plt.subplot(211)
plt.plot(x,E,'g-')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
#print(max(accuracy))