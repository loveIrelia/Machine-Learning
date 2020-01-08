import numpy as np
def loadData(fileName):
    print('start read file')
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为整型
        dataArr.append([int(num) for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        labelArr.append(int(curLine[0]))
    #返回数据集和标记
    return dataArr, labelArr

def getClosest(trainDataMat, trainLabelMat, x, topK):
    len = trainDataMat.shape[0]
    print(trainDataMat.shape)
    x1 = np.tile(x, (len, 1))
    print(x1.shape)
    x1 = (x1 - trainDataMat)**2
    print(x1.shape)
    x2 = np.sum(x1, axis=1)
    topKList = np.argsort(x2)[:topK]
    LabelList = 10*[0]
    for i in topKList:
        LabelList[int(trainLabelMat[i])] += 1
    return LabelList.index(max(LabelList))


def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):

    print('start test')
    trainDataMat = np.array(trainDataArr); trainLabelMat = np.array(trainLabelArr).T
    testDataMat = np.array(testDataArr); testLabelMat = np.array(testLabelArr).T

    errorCnt = 0
    # for i in range(len(testDataMat)):
    for i in range(20):
        # print('test %d:%d'%(i, len(trainDataArr)))
        print('test %d:%d' % (i, 20))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]: errorCnt += 1

    # return 1 - (errorCnt / len(testDataMat))
    return 1 - (errorCnt / 20)

if __name__ == "__main__":
    trainDataArr, trainLabelArr = loadData('mnist_train.csv')
    testDataArr, testLabelArr = loadData('mnist_test.csv')
    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    print('accur is:%d'%(accur * 100), '%')