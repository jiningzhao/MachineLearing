from numpy import *


# 定义层次聚类中结点的类
class cluster_node:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.left = left    # 左结点
        self.right = right  # 右结点
        self.vec = vec      # 特征向量
        self.id = id        # 结点编号
        self.distance = distance    # 距离
        self.count = count  # 结点个数，only used for weighted average


# 定义计算距离的两种方法
def L2dist(v1, v2):
    return sqrt(sum((v1 - v2) ** 2))

def L1dist(v1, v2):
    return sum(abs(v1 - v2))


# 层次聚类算法实现
def hcluster(features, distance=L2dist):        # features是特征矩阵，distance表示使用哪种计算距离的方法
    distances = {}
    currentclustid = -1

    # 初始化，每一个点自成一个类别
    clust = [cluster_node(array(features[i]), id=i) for i in range(len(features))]

    while len(clust) > 1:   # 类别个数>1就一直在执行分类过程，直到只剩下一个类别停止
        # 层次聚类算法：将距离最近的两个类聚集成一类
        lowestpair = (0, 1)         # 比较距离的过程中，先从前两个点开始
        closest = distance(clust[0].vec, clust[1].vec)

        # 遍历所有点，找到距离最小的一对i和j
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # 计算两个点特征的平均值作为新聚类的特征
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 \
                    for i in range(len(clust[0].vec))]

        # 创建新的聚类结点
        newcluster = cluster_node(array(mergevec), left=clust[lowestpair[0]],
                                  right=clust[lowestpair[1]],
                                  distance=closest, id=currentclustid)

        # 更新clust
        currentclustid -= 1         # 生成新聚类后，总的类别个数少了一个
        del clust[lowestpair[1]]    # 删除原来的类别
        del clust[lowestpair[0]]
        clust.append(newcluster)    # 加入新生成的结点

    return clust[0]     # 循环结束后，返回最顶端的结点


# 输入指定的距离dist，得到小于该dist的聚类结果，即得到distance<dist的子树集
def extract_clusters(clust, dist):
    clusters = {}

    if clust.distance < dist:   # 如果clust的根节点的距离就小于dist，返回整棵树
        return [clust]
    else:           # 通过递归的方式，分别判断左子树和右子树是否满足
        cl = []
        cr = []
        if clust.left != None:
            cl = extract_clusters(clust.left, dist=dist)
        if clust.right != None:
            cr = extract_clusters(clust.right, dist=dist)
        return cl + cr


# 获取clust的元素ids
def get_cluster_elements(clust):
    if clust.id >= 0:       # 正数说明是叶子结点，不需要继续
        return [clust.id]
    else:                   # 负数时检查左右分支，递归实现
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)
        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        return cl + cr


# 打印clust
def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print(' ',)
    if clust.id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels == None: print(clust.id)
        else: print(labels[clust.id])


    # now print the right and left branches
    if clust.left != None: printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n + 1)


# 获取树的总高度
def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left == None and clust.right == None: return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left) + getheight(clust.right)

# 获取树的深度
def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left == None and clust.right == None: return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance



