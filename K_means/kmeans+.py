import numpy as np
import matplotlib.pyplot as plt

###创建数据集
def createDataSet():
    DataSet = np.loadtxt('KmeansData.txt', dtype=np.float32)
    return DataSet

###计算欧式距离
def compute_distance(x,y):
    return np.sqrt(np.sum((x - y)**2))

###初始化平均值集合
def randcent(DataSet,k):
    m, n = DataSet.shape
    cent = np.zeros((k,n))
    for i in range (k):
        index = np.random.uniform(0,m)    ###通过均匀分布随机选出k个初始中心点
        cent[i] = DataSet[int(index)]
    return cent

def SSE(DataSet, k):
    loss = 0
    m = np.shape(DataSet)[0]
    mark = np.zeros((m, 1))  #分类集合
    cent = randcent(DataSet, k) #中心集合
    flag = True
    while flag:
        flag = False
        for i in range (m):
            mindistance = np.inf
            miindex = -1
            for j in range(k):
                distance = compute_distance(DataSet[i],cent[j])
                if mindistance > distance:
                    mindistance = distance
                    miindex = j
            if mark[i] != miindex:
                flag = True
            mark[i] = miindex
        for i in range(k):
            choice = DataSet[np.where(mark == i)[0]]
            if len(choice!=0):
                cent[i] = np.mean(choice, axis=0)
    for i in range(k):
        temp  = DataSet[np.where(mark == i)[0]]
        length = temp.shape
        cen = np.tile(cent[i], (length[0], 1))
        temp1 = np.sum(np.sum((temp-cen)**2, axis=1), axis=0)
        loss += temp1
    return  loss

###K-means
def K(DataSet, k):
    m = np.shape(DataSet)[0]
    mark = np.zeros((m, 1))  #分类集合
    cent = randcent(DataSet, k) #中心集合
    flag = True
    while flag:
        flag = False
        for i in range (m):
            mindistance = np.inf
            miindex = -1
            for j in range(k):
                distance = compute_distance(DataSet[i],cent[j])
                if mindistance > distance:
                    mindistance = distance
                    miindex = j
            if mark[i] != miindex:
                flag = True
            mark[i] = miindex
        for i in range(k):
            choice = DataSet[np.where(mark == i)[0]]
            cent[i] = np.mean(choice, axis=0)
    return mark

if __name__ == '__main__':
    DataSet = createDataSet()
    # plt.scatter(DataSet[:, 0], DataSet[:, 1], c='g')
    # plt.show()
    mark = K(DataSet, 4)
    color = ['r', 'g', 'b', 'y']
    for i in range(4):
        temp = DataSet[np.where(mark == i)[0]]
        plt.scatter(temp[:, 0], temp[:, 1], c=color[i])
    plt.show()
    k = np.arange(1,11)
    ssE = []
    for i in range(10):
        ssE.append(SSE(DataSet,k[i]))
    plt.plot(k, ssE)
    plt.show()
