import numpy as np
import pylab as pl     #用于画图的
from sklearn import svm

# 生成训练数据集，40个样本
np.random.seed(0)                         #为了使程序每次生成的随机数相同，填一个参数0，不填参数时每次生成的随机数不同
X = np.r_[np.random.randn(20,2)-[2,2],
          np.random.randn(20,2)+[2,2]]   # 生成训练集，共40个，前20个一组，后20个一组
            #np.r_将一系列的序列合并到一个数组中
            #np.random.randn(20,2)生成标准正态分布，20行2列
            #-[2,2] 和+[2,2] 将样本向左下方和右上方移动，以区分两个类别
Y=[0]*20+[1]*20  #生成标签label，前20个label是0，后20个label是1

#创建模型
clf = svm.SVC(kernel = 'linear')     #创建线性SVM分类器，SVC用于分类问题，SVR用于回归问题
clf = clf.fit(X,Y)                         #创建模型
prediction= clf.predict([(-2,0)])         #预测
print('预测结果:',prediction)


