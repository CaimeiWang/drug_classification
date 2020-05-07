#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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
print('10mg/L',pca_data[1])
print('50mg/L',pca_data[1001])
print('0mg/L',pca_data[2001])
#pca_data=data #不降维
pca_data=pca_data[1000:3000,:]
#制作特征样本和分类标签
pca_data= pca_data.tolist()
for i in range(2000):     #先加入0，1，2标签，制作带标签数据集
    if i<1000:
        pca_data[i].insert(0,0)
    elif i<2000:
        pca_data[i].insert(0,1)
    # elif i<3000:
    #     pca_data[i].insert(0,2)
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
            train_label.append([1,0])
        elif pca_data[i][0]==1:
            train_label.append([0,1])
        # elif pca_data[i][0]==2:
        #     train_label.append([0,0,1])
    else:
        test.append(pca_data[i][1:len(pca_data[i])])
        if pca_data[i][0]==0:
            test_label.append([1,0])
        elif pca_data[i][0]==1:
            test_label.append([0,1])
        # elif pca_data[i][0]==2:
        #     test_label.append([0,0,1])
print(test[1])
# innum=6
# hidnum=12
# outnum=2
# #初始化参数
# w1=np.random.randint(-1,1,(innum, hidnum)).astype(np.float64)
# b1=np.random.randint(-1,1,(1,hidnum)).astype(np.float64)
# w2=np.random.randint(-1,1,(hidnum,outnum)).astype(np.float64)
# b2=np.random.randint(-1,1,(1,outnum)).astype(np.float64)
# w2_1=w2;w2_2=w2_1
# w1_1=w1;w1_2=w1_1
# b1_1=b1;b1_2=b1_1
# b2_1=b2;b2_2=b2_1
# loopNumber=3 #epoch次数
# xite=0.01
# batch_size=3
# accuracy=[]
# E=[]
# FI=[]
# dw1=np.zeros((innum,hidnum),dtype=np.float64) #权重导数
# db1=np.zeros((1,hidnum),dtype=np.float64)#偏差导数
# def sigmoid(x):  #sigmoid函数
#     x=np.array(x)
#     for i in range(x.shape[1]):
#         if x[0][i]>=0:      #对sigmoid函数的优化，避免出现极大的数据溢出
#             x[0][i]=1.0/(1+np.exp(-x[0][i]))
#         else:
#             x[0][i]=np.exp(x[0][i])/(1+np.exp(x[0][i]))
#     return x
# def relu(x):
#     x=np.array(x)
#     for i in range(x.shape[1]):
#         if x[0][i]>0:
#             x[0][i]=1.0/(1+np.exp(-x[0][i]))
#         else:
#             x[0][i]=np.exp(x[0][i])/(1+np.exp(x[0][i]))
#     return x
# def back_relu(x):
#     x=np.array(x)
#     for i in range(x.shape[1]):
#         if x[0][i]>0:
#             x[0][i]=1
#         else:
#             x[0][i]=0
#     return x
# def softmax(z):  #softmax函数用于多分类
#     z_exp=np.exp(z)
#     for i in range(len(z)):
#         z_exp[i]=z_exp[i]/np.sum(z_exp)
#     return z_exp
# def cross_entropy(X,y):   #交叉熵代价函数
#     loss=-(np.multiply(y,np.log(X))+np.multiply((1-y),np.log(1-X)))
#     return loss
# e=0
# for j in range(loopNumber):
#     for i in range(len(train)):
#         input_data=np.mat(train[i])
#         pre_output=np.mat(train_label[i])
#         h=np.dot(input_data,w1)+b1
#         #h_out=sigmoid(h).astype(np.float64) #sigmoid激活函数
#         h_out =relu(h) #relu激活
#         y=np.dot(h_out,w2)+b2
#         #y_out=sigmoid(y).astype(np.float64) #sigmoid激活
#         y_out=relu(y)#relu激活
#         #y_out=softmax(y_out)
#
#         #计算误差并储存
#         #e=pre_output-y_out
#         #e=cross_entropy(y_out,pre_output)
#         e+=(1/(2*outnum))*np.multiply((y_out-pre_output),(y_out-pre_output)) #均方差代价函数
#         if (i+1)%batch_size==0:
#             e=e/batch_size
#             E.append(np.sum(np.abs(e)))#储存损失值画损失函数
#
#             #计算权值变化率
#             db2=e
#             dw2=np.multiply(e,y_out)
#             #FI=np.multiply(h_out,(1-h_out)) #sigmoid
#             FI=back_relu(h_out) #relu
#             for n in range(innum):
#                 for k in range(hidnum):
#                     dw1[n,k]=float(FI[0,k]*float(input_data[0,n])*float((e[0,0]*w2[k,0]+e[0,1]*w2[k,1])))
#                     db1[0,k]=float(FI[0,k])*(e[0,0]*w2[k,0]+e[0,1]*w2[k,1]).astype(np.float64)
#
#             dw1=np.array(dw1)
#             db1=np.array(db1)
#             #更新权重和偏差
#             w1=w1_1+xite*dw1
#             b1=b1_1+xite*db1
#             w2=w2_1+xite*dw2
#             b2=b2_1+xite*db2
#
#             w1_2=w1_1
#             w1_1=w1
#             w2_2=w2_1
#             w2_1=w2
#             b1_2=b1_1
#             b1_1=b1
#             b2_2=b2_1
#             b2_1=b2
#             e=0
#             #储存准确率，以备画准确率曲线
#             result = []
#             for i in range(len(test)):
#                 input_data = np.mat(test[i])
#                 pre_output=np.mat(test_label[i])
#                 h=np.dot(input_data,w1_2)+b1_2
#                 #h_out=sigmoid(h) #sigmoid激活函数
#                 h_out =relu(h)#relu激活
#                 y=np.dot(h_out,w2_2)+b2_2
#                 #y_out=sigmoid(y)#sigmoid激活
#                 y_out =relu(y)#relu激活
#                 y_out=softmax(y_out)
#                 if np.argmax(y_out)==np.argmax(pre_output):
#                     result.append(1)
#                 else:
#                     result.append(0)
#             pre_result=np.sum(result)/len(result)
#             print(pre_result)
#             accuracy.append(pre_result)
#         else:
#             continue
#         # if pre_result>0.9:
#         #     break
# # np.save('accuracy.npy',accuracy)
# # np.save('w1.npy',w1)
# # np.save('w2.npy',w2)
# # np.save('b1.npy',b1)
# # np.save('b2.npy',b2)
# #print(w1,w2,b1,b2)
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