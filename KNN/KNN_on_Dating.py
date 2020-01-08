import numpy as np
import matplotlib.pyplot as plt
import os
import operator

def file2matrix(filename):
    """
    函数说明：加载数据集
    parameters:
        fileName - 文件名
    return:
        featureMat - 特征矩阵
        classLabelVector - 类别标签向量(didntLike - 0, smallDoses - 1, largeDoses - 2)
    """
    f = open(filename)
    featureMat = []
    classLabelVector = []
    for line in f.readlines():
        curline = line.strip().split('\t')
        featureMat.append([float(num) for num in curline[:3]])
        classLabelVector.append(curline[3])
    featureMat = np.array(featureMat)
    classLabelVector = np.array(classLabelVector)
    return featureMat, classLabelVector

def autoNorm(dataSet):
    """
    函数说明：数据归一化
    parameters:
        dataSet - 特征矩阵
    return:
        normDataset - 归一化特征矩阵
        ranges - 数据范围(Xmax - Xmin)
        minVals - 最小特征值
    """
    length = dataSet.shape[1]
    for i in range (length):
        max_ = max(dataSet[:,i])
        min_ = min(dataSet[:,i])
        dataSet[:,i]= (dataSet[:,i]-min_)/(max_-min_)
    return dataSet

def kNNClassify(inX, dataSet, labels, k):
    """
    函数说明：kNN分类
    parameters:
        inX - 用于要进行分类判别的数据(来自测试集)
        dataSet - 用于训练的数据(训练集)
        labels - 分类标签
        k - kNN算法参数，选择距离最小的k个点
    return:
        predLabel - 预测类别
    """
    length = dataSet.shape[0]
    inX = np.tile(inX,(length,1))
    inX = np.sum((inX-dataSet)**2,axis=1)
    indexvec = np.argsort(inX)[:k]
    count={}
    for i in range(k):
        count[labels[indexvec[i]]] = count.get(labels[indexvec[i]],0)+1
    return max(count,key=count.get)


def datingClassTest():
    """
    函数说明：测试kNN分类器

    """
    alpha = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')  # load data setfrom file
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * alpha)
    errorCount = 0.0
    for i in range(numTestVecs):
        key = kNNClassify(datingDataMat[i, :], datingDataMat[numTestVecs:, ], datingLabels[numTestVecs:], 4)
        if key != datingLabels[i]:
            errorCount += 1.0
    print("The error is : %f" % errorCount)

if __name__ == '__main__':
    datingClassTest()