from sklearn.cluster import KMeans
import numpy as np

X=np.array([[1,1],[2,1],[4,3],[5,4]])        # 样本
kmean = KMeans(n_clusters=2,max_iter=10)    # 构造kMeans算法模型
kmean.fit(X)   #让模型开始分类
print(kmean.labels_)   # 获取分类结果
# 非监督学习，之前不知道样本label
X1=[[1.5,0]]
Y1=kmean.predict(X1)  # 预测
print(Y1)