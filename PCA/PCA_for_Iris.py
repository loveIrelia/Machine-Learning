from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
data = load_iris()
data1 = data.data
target = data.target
m, n = data1.shape
# 计算每个特征的均值
ave = np.mean(data1, axis=0)
# 扩展均值矩阵
ave_mat = np.tile(ave, [m, 1])
# 去中心化
data1 -= ave_mat
# 计算协方差矩阵
co = np.cov(data1.T)
# 特征值分解
e_vals, e_vecs = np.linalg.eig(co)
# 降序排序（最大特征值所对应的索引）
e_vals_index = np.argsort(e_vals)[::-1]
# 特征向量矩阵
a = np.array([e_vecs[:, e_vals_index[0]], e_vecs[:, e_vals_index[1]],e_vecs[:, e_vals_index[2]]])
# 降维后的数据集
pca_mat = np.dot(data1, a.T)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})

label = [0, 1, 2]
color = ['r', 'g', 'b']

for i in range(len(label)):
    temp = pca_mat[np.where(target == label[i])[0]]
    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c=color[i])
plt.show()
