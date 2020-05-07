from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import learning_curve

def plot_PR(model, x_test, y_test):  # 绘制PR曲线
    y_pro = model.predict_proba(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pro[:, 1])
    average_precision = average_precision_score(y_test, y_pro[:, 1])
    ax2 = plt.subplot(224)
    ax2.set_title("Precision_Recall Curve AP=%0.2f" % average_precision, verticalalignment='center')
    plt.step(precision, recall, where='post', alpha=0.2, color='r')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')


def plot_ROC(model, x_test, y_test):  # 绘制ROC和AUC，来判断模型的好坏
    y_pro = model.predict_proba(x_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_pro[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    ax3 = plt.subplot(223)
    ax3.set_title("Receiver Operating Characteristic", verticalalignment='center')
    plt.plot(false_positive_rate, recall, 'b', label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('false_positive_rate')
    plt.show()

#roc
for i in range(len(test)):
    test[i]=np.array([test[i]])

plot_ROC(model,np.array(test),np.array(test_label1))