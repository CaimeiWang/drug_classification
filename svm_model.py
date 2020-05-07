#加载需要用到的包
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from keras.optimizers import *
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import learning_curve
from sklearn import svm
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
            val_label.append(0)
        elif pca_data[i][0]==1:
            val_label.append(1)
        elif pca_data[i][0]==2:
            val_label.append(2)
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

#训练
# c_count=[]
# accuracy=[]
# n_support_count=[]
# for i in range(1,11):
#     b=i/10
#     #print(c)
#     c_count.append(b)
#     clf=svm.SVC(C=b)#惩罚系数C越大，对分错样本的惩罚越大，训练样本的准确率越高。但是泛化能力降低。
#     clf.fit(train,train_label)
#     n_support_count.append(clf.n_support_)
#     #print(clf.n_support_)
#     #测试
#     y_true=test_label1
#     y_pred=clf.predict(test)
#     #print(y_pred)
#     a=[]
#     for i in range(len(y_pred)):
#         if y_pred[i]==y_true[i]:
#             a.append(1)
#         else:
#             a.append(0)
#     accuracy.append(sum(a)/len(a))
# #print(accuracy)
# plt.plot(c_count,accuracy,'g-')
# plt.xlabel('惩罚因子')
# plt.ylabel('准确率')
# plt.show()

clf=svm.SVC(C=1)
clf.fit(train,train_label)
y_true=test_label1
y_true1=test_label
y_pred=clf.predict(test)
y_pred1=[]
for i in range(len(y_pred)):
    if y_pred[i]==0:
        y_pred1.append([1,0,0])
    elif y_pred[i]==1:
        y_pred1.append([0,1,0])
    elif y_pred[i]==2:
        y_pred1.append([0,0,1])

np.save('y_true_svm.npy',y_true1)
np.save('y_pred_svm.npy',y_pred1)
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


