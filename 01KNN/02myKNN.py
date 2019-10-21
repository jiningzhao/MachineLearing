# 自己编码实现KNN算法，给花进行分类
import math
import csv    #逗号分隔的文件可以用csv格式解释
import random
import operator
# 计算距离的函数
def CalclateDistance(n,k,length):
    s=0
    for i in range(length):
        s+=pow((n[i]-k[i]),2)
    return math.sqrt(s)

# a=[6.0,2.2,4.0,1.0]
# b=[5.8,2.7,4.1,1.0]
# print(CalcDistance(a,b))

#加载数据集，将数据分为训练集和测试集两部分，比例通过参数传递
def loadDataset(filename,split,trainingSet=[],testSet=[]):
# split是训练集所占比重，trainingSet是解析后得到的训练集，testSet是解析后得到的测试集
     with open(filename,'rt') as csvfile:
         lines=csv.reader(csvfile)             #读取文件有多少行
         dataSet=list(lines)                   #将数据转化为list，列表中每一行就是一个样本
         #将dataset里前4列由字符串转成float，用于后面计算距离
         for x in range(len(dataSet)):
             for y in range(4):
                 dataSet[x][y] = float(dataSet[x][y])         #特征转成float
            #将数据分配到训练集和测试集
             if random.random() < split:
                 trainingSet.append(dataSet[x])
             else:
                 testSet.append(dataSet[x])
#返回距离最近的k个邻居，trainingSet是训练样本集合，testInstance是待测试样本
def getNeighbors(trainingSet,testInstance,k):
    distance=[]

    for i in range(len(trainingSet)):
        d=CalclateDistance(trainingSet[i],testInstance, len(testInstance)-1)     #计算测试样本和所有训练集点的距离
                                                                            #样本最后一个参数是类别标签，不需要参与距离的运算
        distance.append((trainingSet[i],d))                            #将数据样本和距离绑在一起，方便后续排序
    #print(distance[0])  #得到结果为([5.1,3.5,1.4,0.2,'Iris-setosa'],0.509901951359278)
    distance.sort(key=operator.itemgetter(1))                        #import operator，按第一维的距离从小到大排序

    return distance[0:k]

# 根据邻近点的列表进行评测
def getClassify(neighbor):
    classVotes={}    #创建一个字典存放类别
    for i in range(len(neighbor)):
        tempClass=neighbor[i][0][-1]          #找到记录类别的位置
        if tempClass in classVotes:
            classVotes[tempClass]+=1
        else:
            classVotes[tempClass]=1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)                     #从大到小排序
    return sortedVotes[0][0]         #返回个数最多的类别

#计算准确率
def getAccuracy(testSet,predictions):    #testSet是测试集，predictions是对测试集进行kNN算法得到的预测值
    correct=0
    for i in range(len(testSet)):
        if testSet[i][-1]==predictions[i]:       #若真实值==预测值
            correct+=1
    return (correct/len(testSet))

#--------------------------------------------------------------------------
#读取数据集的文件并将数据分为训练集和测试集
filename="../datasets/irisdata.txt"
trainingSet=[]    #训练集
testSet=[]        #测试集
loadDataset(filename,0.67,trainingSet,testSet)
print("train set:",len(trainingSet))
print("test set:",len(testSet))

#通过KNN算法预测测试集的数据分类
k=3
predictions=[]        #用来存放预测结果
for i in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[i], k)
    predictions.append(getClassify(neighbors))
accuracy=getAccuracy(testSet,predictions)

print("准确率:",accuracy)