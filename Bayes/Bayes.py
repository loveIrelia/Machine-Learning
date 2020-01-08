import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def CreateDataSet():
    """
    返回男女数据集(列表)
    :return:
    """
    data = pd.read_excel('2.xlsx')
    DataSet = data.values
    DataSet = DataSet.tolist()
    boys_data = []; girls_data=[]
    for i in range(len(DataSet)):
        if(DataSet[i][3]==1):
            boys_data.append(DataSet[i])
        else:
            girls_data.append(DataSet[i])
    return boys_data, girls_data

def Divide_DataSet(num1, num2, boys_data, girls_data):
    """
    划分数据集，形成训练集和测试集
    :param P:
    :param num:
    :param boys_data:
    :param girls_data:
    :return:
    """
    test = []
    for i in range(num1):
        index1 = np.random.randint(len(girls_data))
        test.append(girls_data[index1])
        girls_data.pop(index1)

    for j in range(num2):
        index2 = np.random.randint(len(boys_data))
        test.append(boys_data[index2])
        boys_data.pop(index2)
    test = np.array(test)
    len_boy = len(boys_data)
    len_girl = len(girls_data)
    p_boys = len_boy / (len_boy + len_girl)
    p_girl = len_girl / (len_girl + len_boy)
    return test,p_boys,p_girl

def Gauss_Distribution(data):
    """
    返回一维高斯分布的参数
    :param data:
    :return:
    """
    mean = np.mean(data)
    var = np.var(data)
    return mean, var

def Bayes(means_boys, var_boys, mean_girls, var_girls, p_boy,p_girl, vec):
    """
    计算概率，判断男女
    :param means_boys:
    :param var_boys:
    :param mean_girls:
    :param var_girls:
    :param p:
    :param vec:
    :return:
    """
    p1 = stats.norm.pdf(vec, means_boys, var_boys)*p_boy
    p2 = stats.norm.pdf(vec, mean_girls, var_girls)*p_girl
    p_1 = p1/(p1+p2)
    p_2 = p2/(p1+p2)
    if p_1 > p_2:
        return p_1, 1
    else:
        return p_1, 0

def N_dimension_Gauss_Distribution(data, k1, k2):
    """
    返回多维高斯分布的均值向量和协方差矩阵
    :param data:
    :return:
    """
    data1 = data[:, [k1, k2]].T
    mean = np.mean(data1, axis=1)
    cov = np.cov(data1)
    return mean, cov
def change_cov(cov):
    m,n = cov.shape
    temp = np.zeros((m, n))
    for i in range(m):
        temp[i, i] = cov[i, i]
    return temp
def N_dismension_Bayes(mean_boys, cov_boys, mean_girls, cov_girls, p_boy, p_girl ,vec):
    """
    计算概率，判断男女
    :param mean_boys:
    :param cov_boys:
    :param mean_girls:
    :param cov_girls:
    :param p:
    :param vec:
    :return:
    """
    p1 = multivariate_normal.pdf(vec, mean_boys, cov_boys)*p_boy
    p2 = multivariate_normal.pdf(vec, mean_girls, cov_girls)*p_girl
    p_1 = p1/(p1+p2)
    p_2 = p2/(p1+p2)
    if p_1 > p_2:
        return p_1, 1
    else:
        return p_1, 0
def Parzen(alldata, x, h):
    """

    :param alldata:
    :param x:
    :param h:
    :return:
    """
    n = alldata.shape[0]
    q = 0
    for j in range(n):  # 遍历每个样本
        if abs((alldata[j] - x) / h) <= 0.5:  # 方窗
            q += 1
    return (q/n)/h
# def Create_Parzen_Plot(b, test):
#     x = []
#     y = []
#     t = np.sort(test[:, 0])
#     for i in range(test[:, 0].shape[0]):
#         y.append(Parzen(b, t[i], 0.5))
#         x.append(t[i])
#     plt.plot(x, y)
#     plt.show()
def ROC(prediction,test,trueclass,flaseclass):
    """
    返回TPR 和 FPR
    :param prediction:
    :param test:
    :param trueclass:
    :param flaseclass:
    :return:
    """
    TP = 0  # 预测真为真
    FN = 0  # 预测真为假
    FP = 0  # 预测假为真
    TN = 0  # 预测假为假
    for i in range(test.shape[0]):
        if prediction[i] == trueclass and test[i,3] == trueclass:
            TP = TP + 1
        elif prediction[i] == flaseclass and test[i,3] == flaseclass:
            TN = TN + 1
        elif prediction[i] == trueclass and test[i,3] == flaseclass:
            FP = FP + 1
        elif prediction[i] == flaseclass and test[i,3] == trueclass:
            FN = FN + 1

    # if (FP + TN) == 0:  # 因为样本选取可能没有假例，此时预测假为假的TN值永远为零，当FP也为零时，分母为零，报错，这样做为了防止报错，因此最好不要用只有一类的样本集
    #     return 0, TP / (TP + FN)
    return FP / (FP + TN), TP / (TP + FN)

def accuracy(predicted, test):
    """
    计算准确率
    :param predicted:
    :param test:
    :return:
    """
    count = 0
    size = test.shape[0]
    for i in range(size):
        if test[i, 3] == predicted[i]:
            count += 1
    return count/size
def practice_of_parzen(n1, n2, index):
    """

    :param n1:
    :param n2:
    :param i:
    :return:
    """
    boys_data, girls_data = CreateDataSet()
    predict = []  # 预测列表
    test, p_boy, p_girl  = Divide_DataSet(n1, n2, boys_data, girls_data)
    boys_data = np.array(boys_data)
    girls_data = np.array(girls_data)
    for i in range(test.shape[0]):
        p1 = Parzen(boys_data[:, index], test[i, index], 0.6)*p_boy
        p2 = Parzen(girls_data[:, index], test[i, index], 0.6)*p_girl
        if p1 > p2:
            predict.append(1)
        else:
            predict.append(0)
    # print('准确率为:{0}%'.format(accuracy(predict, test) * 100))
    return accuracy(predict, test) * 100
def practice_of_one_demension(n1,n2,i):
    """

    :param n1: 随机女生测试人数
    :param n2: 随机男生测试人数
    :param i: 某一个特征
    :return:
    """
    boys_data, girls_data= CreateDataSet()
    predict = []  # 预测列表
    test, p_boy, p_girl  = Divide_DataSet(n1, n2, boys_data, girls_data)
    boys_data = np.array(boys_data)
    girls_data = np.array(girls_data)
    mean_boy, var_boy = Gauss_Distribution(boys_data[:, i])
    mean_girl, var_girl = Gauss_Distribution(girls_data[:, i])
    for k in range(len(test)):
        c1, label = Bayes(mean_boy, var_boy, mean_girl, var_girl, p_boy, p_girl, test[k, i])
        predict.append(label)
    x = []
    y = []
    for k in np.arange(0, 1, 0.01):
        predicted = []
        for j in range(len(test)):
            c1, label = Bayes(mean_boy, var_boy, mean_girl, var_girl, p_boy, p_girl, test[j, i])
            if c1> k:
                predicted.append(1)
            else:
                predicted.append(0)
        r1, r2 = ROC(predicted, test, 1, 0)
        x.append(r1)
        y.append(r2)
    x = np.array(x)
    y = np.array(y)
    plt.plot(x, y)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.show()
    # print('准确率为:{0}%'.format(accuracy(predict, test)*100))
    return accuracy(predict, test)*100
def practice_of_multiple_demension(k1, k2):
    """
    用两个特征来判断男女
    :return:
    """
    boys_data, girls_data = CreateDataSet()
    test, p_boy, p_girl = Divide_DataSet(5, 10, boys_data, girls_data)
    boys_data = np.array(boys_data)
    girls_data = np.array(girls_data)
    mean_boy, cov_boy1 = N_dimension_Gauss_Distribution(boys_data, k1, k2)
    mean_girl, cov_girl1 = N_dimension_Gauss_Distribution(girls_data, k1, k2)
    cov_boy = change_cov(cov_boy1)
    cov_girl = change_cov(cov_girl1)
    x = []
    y = []
    predicted = []  # 预测列表
    for i in range(len(test)):
        temp = test[i, [k1, k2]]
        c1, c2 = N_dismension_Bayes(mean_boy, cov_boy, mean_girl, cov_girl, p_boy, p_girl, temp.T)
        predicted.append(c2)
    for j in np.arange(0, 1, 0.01):
        predict =[]
        for i in range(len(test)):
            temp = test[i, [k1, k2]]
            c1, c2 = N_dismension_Bayes(mean_boy, cov_boy, mean_girl, cov_girl, p_boy, p_girl, temp.T)
            if c1 > j:
                predict.append(1)
            else:
                predict.append(0)
        r1, r2 = ROC(predict, test, 1, 0)
        x.append(r1)
        y.append(r2)
    plt.plot(x, y)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.show()
    # print('准确率为:{0}%'.format(accuracy(predicted, test) * 100))
    return accuracy(predicted, test) * 100

def Compare(n1,n2,n3):
    """

    :param n1: 特征1
    :param n2: 特征2
    :param n3: 特征3
    :return: 返回三种统计方法的正确率
    """
    x1 = [];x2 = [];x3 = []
    for i in range(10):
        x1.append(practice_of_parzen(5, 10, n3))
        x2.append(practice_of_one_demension(5, 10, n3))
        x3.append(practice_of_multiple_demension(n1, n2))
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    print('Parzen窗法的准确率为:{:.2f}%'.format(np.sum(x1)/10))
    print('一维贝叶斯的准确率为:{:.2f}%'.format(np.sum(x2)/10))
    print('二维贝叶斯的准确率为:{:.2f}%'.format(np.sum(x3)/10))

if __name__ =='__main__':
    # practice_of_multiple_demension(0, 1)
    # practice_of_one_demension(5,10,0)
    # practice_of_parzen(5, 10, 0)
    Compare(1, 2, 1)


