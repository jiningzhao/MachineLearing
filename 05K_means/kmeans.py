import numpy as np


# K Means 算法实现
def kmeans(X, k, maxIt):    # X是数据集，k是分为几类，maxIt是最大迭代次数
    numPoints, numDim = X.shape     # 获取X的行数和列数

    dataSet = np.zeros((numPoints, numDim + 1)) # 定义dataSet比X多一列，将判决出的类别放在多出的这一列中
    dataSet[:, :-1] = X

    # 初始化中心点
    centroids = dataSet[np.random.randint(numPoints, size=k), :]    # 随机选取k个中心点
    centroids = dataSet[0:2, :]     # 重新初始化中心点为前两个点，为了和课件保持一致。正常算法没有这一行代码
    centroids[:, -1] = range(1, k + 1)  # 中心点的标签定为1~k

    # Initialize book keeping vars.
    iterations = 0          # 迭代次数
    oldCentroids = None     # 上一次迭代的中心点

    # k-means算法迭代
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):   # 判断迭代是否需要结束
        print("iteration: ", iterations)
        print("dataSet: \n", dataSet)
        print("centroids: \n", centroids)

        # 保存当前中心点及迭代次数
        oldCentroids = np.copy(centroids)   # 直接用等号的话oldCentroids和centroids占用同一片内存，一会儿会更新centroids，所以这里要复制一份
        iterations += 1

        # 根据中心点，为数据集匹配标记
        updateLabels(dataSet, centroids)

        # 根据新的类别和k值，更新中心点
        centroids = getCentroids(dataSet, k)

    return dataSet


# 判断迭代是否需要停止，oldCentroids==centroids，或iterations>maxIt时停止迭代
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)


# 根据中心点，为数据集匹配标记
def updateLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    numPoints, numDim = dataSet.shape   # 获取dataSet的行数和列数
    for i in range(0, numPoints):       # 为每一个样本匹配类别标记
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)

# 根据距离哪个中心点最近决定样本的类别标记
def getLabelFromClosestCentroid(dataSetRow, centroids):         #dataSetRow一个样本点，centroids所有中心点
    label = centroids[0, -1]       # 暂定传入样本的label是第一个中心点的label
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])    # 以第一个中心点的距离作为比较的依据，linalg.norm计算两个向量的距离
    for i in range(1, centroids.shape[0]):  # 循环i是针对第几个中心点，centroids.shape[0]中心点有几行
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])     # 样本点和第i个中心点的距离
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]   # 将样本l点的abel更新为第i个中心点的label

    print("minDist:", minDist)
    return label


# 更新中心点
def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))   # dataSet.shape[1]表示样本点的列数
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]      # 找出所有label为i的样本，放到oneCluser里，最后一列label不要
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)    # axis=0对每一列求均值，输出矩阵是一行的
        result[i - 1, -1] = i                               # 新的中心点的列表标记

    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print("final result:", result)

