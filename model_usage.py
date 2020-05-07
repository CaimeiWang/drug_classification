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
from sklearn import metrics
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', 100000)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))   # 0.2是神经元舍弃比
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation ='softmax'))

x1=np.array([[-3.39978982,-4.38635045,  0.02212197,  0.01030971,  0.22381213, -1.1313619,
 -1.36080376,  0.62957983, -2.06080212, -0.76918012,  1.31936039, -0.58639637,
 -0.3779383,   0.1394947,   0.11175505,  1.25359998, -0.56669543,  0.28622757,
  0.36966767, -0.47769733, -0.31860586,  0.54596781,  0.07301221,  0.11377999,
 -0.12022339,  0.01321615,  0.73537826, -0.80498586,  0.11015291, -0.658002]]) #待测样本1:10

x2=np.array([[-1.07371511e+01,  4.66260246e-01,  6.91014437e+00,  5.34760986e+00,
  4.35057617e-01, -1.38421832e+00, -1.32889050e+00,  1.28057643e+00,
 -7.80360899e-01, -1.41846377e-01,  2.29311510e-02, -8.59350828e-02,
  1.79576629e-01,  3.21912457e-01, -4.52940162e-02,  3.10540960e-03,
 -9.79429788e-02,  3.44197745e-01,  1.71140891e-02, -2.19296954e-01,
  1.80554262e-01, -3.31620312e-01, -7.91776173e-03, -1.61324006e-01,
 -1.24401251e-01, -1.19711728e-02, -1.54445824e-01,  1.99677563e-02,
 -1.17376149e-01, -2.35998490e-01]]) #待测样本2:50

x3=np.array([[-5.40710542,  0.76423273, -0.10785392, -1.39571763,  1.66013203, -1.02748131,
 -0.42673462, -1.52596927, -0.33318346, -0.42383635,  0.28210704, -0.28246292,
 -0.33281097,  0.71218231,  0.2121216,   0.07275207,  0.3892159,   1.05428991,
 -0.12011813, -0.13485165,  0.18013537, -0.01347465,  0.04441498, -0.23977776,
  0.16737721,  0.15888478,  0.31984838, 0.53078465, -0.06003662,  0.08475696]])#待测样本2:0


#载入网络参数
model.load_weights('drug_classify_model1.h5')
# category=model.predict_classes(x3)
# if category==0:
#     print('待检测毒品的密度为：10mg/L')
# elif category==1:
#     print('待检测毒品的密度为：50mg/L')
# elif category==2:
#     print('待检测毒品的密度为：0mg/L')

def confusion_metrix(y,y_p):
    Confusion_matrix=confusion_matrix(y,y_p) #y代表真实值，y_p 代表预测值
    plt.matshow(Confusion_matrix)
    plt.title("混淆矩阵")
    plt.colorbar()
    plt.ylabel("实际类型")
    plt.xlabel("预测类型")
    plt.show()


if __name__ == '__main__':
    test=np.load('test.npy')
    test_label= np.load('test_label.npy')
    test_label1=np.load('test_label1.npy')

    #混淆矩阵
    y_pre=[]
    y_pre1=[]
    for i in range(len(test)):
        pre=model.predict_classes(np.array([test[i]]))
        y_pre.append(pre[0])
        if pre==0:
            y_pre1.append([1,0,0])
        elif pre==1:
            y_pre1.append([0,1,0])
        if pre==2:
            y_pre1.append([0,0,1])
    #confusion_metrix(test_label1,y_pre)

    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # test_label=np.array(test_label)
    # y_pre1=np.array(y_pre1)
    # for i in range(3):
    #     fpr[i], tpr[i], _ = roc_curve(test_label[:, i], y_pre1[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Plot of a ROC curve for a specific class
    # for i in range(3):
    #     plt.figure()
    #     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic example')
    #     plt.legend(loc="lower right")
    #     plt.show()
    #DNN
    test_label=np.array(test_label)
    y_pre1=np.array(y_pre1)
    fpr, tpr, thresholds = metrics.roc_curve(test_label.ravel(), y_pre1.ravel())
    auc = metrics.auc(fpr, tpr)

    #SVM
    y_pre_svm=np.load('y_pred_svm.npy')
    y_true_svm=np.load('y_true_svm.npy')
    y_pre_svm=np.array(y_pre_svm)
    y_true_svm=np.array(y_true_svm)
    # print(y_pre_svm)
    # print(y_true_svm)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_true_svm.ravel(),y_pre_svm.ravel())
    auc1= metrics.auc(fpr1, tpr1)
    #bays
    y_pre_bays=np.load('y_pred_bays.npy')
    y_true_bays=np.load('y_true_bays.npy')
    y_pre_bays=np.array(y_pre_bays)
    y_true_bays=np.array(y_true_bays)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true_bays.ravel(), y_pre_bays.ravel())
    auc2= metrics.auc(fpr2, tpr2)

    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c='r', lw=1, alpha=0.7, label=u'深度神经网络的AUC=%.3f' % auc)
    plt.plot(fpr1, tpr1, c='g', lw=1, alpha=0.7, label=u'支持向量机的AUC=%.3f' % auc1)
    plt.plot(fpr2, tpr2, c='y', lw=1, alpha=0.7, label=u'朴素贝叶斯的AUC=%.3f' % auc2)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC and AUC', fontsize=17)
    plt.show()

