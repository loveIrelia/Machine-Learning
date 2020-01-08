import numpy as np
import matplotlib.pyplot as plt
data1=np.loadtxt('PCAData.txt')
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
print(e_vals_index)
a = e_vecs[:, e_vals_index[0]]
print(a)
# 降维后的数据集
pca_mat = np.dot(data1, a)
print(pca_mat.shape)
# plt.scatter(pca_mat[:, 0], pca_mat[:, 0])
# plt.show()