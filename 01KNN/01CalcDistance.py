import math

def CalulateDistance(x1,y1,x2,y2):
    d=math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return d

d_1=CalulateDistance(3,104,18,90)
d_2=CalulateDistance(2,100,18,90)
d_3=CalulateDistance(1,81,18,90)
d_4=CalulateDistance(101,10,18,90)
d_5=CalulateDistance(99,5,18,90)
d_6=CalulateDistance(98,2,18,90)

print("d_1:",d_1)
print("d_2:",d_2)
print("d_3:",d_3)
print("d_4:",d_4)
print("d_5:",d_5)
print("d_6:",d_6)