import math
import numpy as np
#求x,y的分别的平均数xn,yn
def A(x,y):
    sum,sum1=0,0
    for i in range(len(x)):
        sum+=x[i]
    xn=sum/len(x)
    for i in range(len(y)):
        sum1+=y[i]
    xn=sum/len(x)
    yn=sum1/len(y)
    return xn,yn
#求B1
def B(xn,yn,x,y):
    sum=0
    sum1=0
    for i in range(len(x)):
        sum+=(x[i]-xn)*(y[i]-yn)
    for i in range(len(x)):
        sum1+=pow((x[i]-xn),2)
    B1=sum/sum1
    return B1
#求B0
def C(B1,xn,yn):
    return yn-xn*B1
#测试
def D(x,B1,B0):
    return B1*x+B0

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
xn,yn=A(x,y)
B1=B(xn,yn,x,y)
B0=C(B1,xn,yn)
print("估计方程为：","y=",B1,"x+",B0)

x1=float(input("请输入x值："))
print("y值为：",D(x1,B1,B0))


def fitSLR(x,y):   #创建简单线性回归模型
    sumX,sumY,sum1,sum2=0,0,0,0
    for i in range(len(x)):
        sumX+=x[i]
        sumY+=y[i]
    pX,pY=sumX/len(x),sumY/len(y)
    for i in range(len(x)):
        sum1+=(x[i]-pX)*(y[i]-pY)
        sum2+=pow((x[i]-pX),2)
    B1=sum1/sum2
    B0=pY-B1*pX
    return B1,B0
def pridict(B1,B0,x):
    Y=B1*x+B0
    return Y
x11 = [1, 3, 2, 1, 3]
y11 = [14, 24, 18, 17, 27]
k1,k2=fitSLR(x11,y11)
xk=int(input("请输入x:"))
print("Y值为:",pridict(k1,k2,xk))

