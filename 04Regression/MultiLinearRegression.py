from numpy import genfromtxt
from sklearn import linear_model

#读取数据
dataPath='../datasets/Delivery.csv'
data=genfromtxt(dataPath,delimiter=',')     #从文件中读取数据，并转换成numpy.array格式
print(data)


X=data[:,:-1]   #训练数据集
Y=data[:,-1]   #标记


regr=linear_model.LinearRegression()    #1.创建线性回归模型
regr.fit(X,Y)                           #2.模型开始训练
xPre=[[102,6]]
yPre=regr.predict(xPre)
print('预测结果：',yPre)

print("b1,b2:",regr.coef_)                 #自变量的系数，也叫权重
print("b0:",regr.intercept_)               #偏置项