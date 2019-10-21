from sklearn import neighbors
from sklearn import datasets     # 已有的数据包括iris数据集


iris=datasets.load_iris()                 #2.加载iris数据集
print(iris)

knn=neighbors.KNeighborsClassifier()      #1.创建KNN分类器
knn.fit(iris.data,iris.target)            #3.建立KNN模型，第一个参数是训练集的特征，第二个参数是训练集的标签label
prediction=knn.predict([[0.1,0.2,0.3,0.4]])   #4.用第三步建立好的模型进行预测
print(prediction)