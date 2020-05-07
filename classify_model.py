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
            train_label.append([1,0,0])
        elif pca_data[i][0]==1:
            train_label.append([0,1,0])
        elif pca_data[i][0]==2:
            train_label.append([0,0,1])
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
np.save('test.npy',test)
np.save('test_label.npy',test_label)
np.save('test_label1.npy',test_label1)
#构建网络
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))   # 0.2是神经元舍弃比
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation ='softmax'))
model.summary() #显示网络

#编译网络结构
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

#训练网咯
epochs=20
batch_size=32
history = model.fit(train,train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, #0，不显示，1，进度条，2，文字
                    validation_data=(val,val_label))


def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    #plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    #plt.legend(['train', 'validation'])
    plt.legend(['validation', 'train'])


# 显示训练过程
def plot(history):
    plt.figure(figsize=(6.5,6))
    plt.subplot(2,1,1)
    show_train_history(history,'loss','val_loss')
    plt.subplot(2,1,2)
    show_train_history(history, 'accuracy', 'val_accuracy')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

plot(history)
score = model.evaluate(test,test_label,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

def confusion_metrix(y,y_p):
    Confusion_matrix=confusion_matrix(y,y_p) #y代表真实值，y_p 代表预测值
    plt.matshow(Confusion_matrix)
    plt.title("混淆矩阵")
    plt.colorbar()
    plt.ylabel("实际类型")
    plt.xlabel("预测类型")


#保存模型
model.save_weights('drug_classify_model1.h5')