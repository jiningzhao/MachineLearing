# 以课件中决策树的例子讲解ID3决策树算法

import csv
from sklearn.feature_extraction import DictVectorizer        # 决策树处理的数据需要转换成指定格式
from sklearn import preprocessing
from sklearn import tree

labelList=[]         #标签
featureList=[]        #特征值


#1. 从文件中读取数据
with open("../datasets/AllElectronics.csv","rt") as csvfile:
    reader = csv.reader(csvfile)
    headers=next(reader)    # headers是文件第一行

    for row in reader:                    #从文件中逐行读取样本数据
        labelList.append(row[-1])         #标签放在最后一列
        rowDict={}
        for i in range(1,len(row)-1):     #每行第一个序号没用，最后一个是标签也不用
            rowDict[headers[i]]=row[i]
        featureList.append(rowDict)
    print(featureList)
    print(labelList)

#1.对特征数据featureList进行格式转换
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()   #对特征进行调整
print('调整后特征名:',vec.get_feature_names())     #格式调整后的特征名
print('dummyX:',dummyX)

#2.对标签label进行格式转换
label=preprocessing.LabelBinarizer()
dummyY=label.fit_transform(labelList)            #将label转化成指定的格式
print('dummyY:',dummyY)

#3. 创建决策树
clf=tree.DecisionTreeClassifier(criterion='entropy')    # 创建决策树分类器，entropy表示使用信息熵算法
clf = clf.fit(dummyX,dummyY)                            # 创建决策树模型,第一个参数是训练集的特征，第二个参数是训练集的标签label

#4. 预测
new=dummyX[0,:]
new[0]=1              #将原来数据中age从yang改成middle
new[2]=0
prediction= clf.predict([new])
print('预测结果:',prediction)

#5. 将决策树输出到文件中,用graphviz工具画出决策树的图
with open('DTree.dot','w') as file:
    file = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=file)
    #第一个参数：决策树模型
    #feature_names:用于将特征值还原成原始字符串的形式
    # 安装graphviz后，在环境变量中加入graphviz软件bin文件夹的路径
    # 在命令行中进入dot文件的路径，输入dot -Tpdf DTree.dot -o output.pdf，生成的output.pdf文件就是决策树模型
