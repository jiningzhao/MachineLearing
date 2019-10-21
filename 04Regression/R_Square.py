import numpy as np


def polyfit(x,y,degree):             #degree是多项式拟合方程中x指数的最大值
    retults={}                       #字典


    # 多项式拟合
    coeffs=np.polyfit(x,y,degree)    #coeffs是得到的方程系数
    print(coeffs)

    # 给定x，计算其预测值yhat
    p=np.poly1d(coeffs)              #以coeffs作为参数生成一个最高一次幂的拟合模型
    yhat=p(x)

    # 计算R-Squared
    y_mean=np.mean(y)
    ssr=np.sum((yhat-y_mean)**2)
    sst=np.sum((y-y_mean)**2)
    r_Square=ssr/sst
    return r_Square

X=[1,3,8,7,9]
Y=[10,12,24,21,34]

print(polyfit(X,Y,1))
