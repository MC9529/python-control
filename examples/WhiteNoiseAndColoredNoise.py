import os
import numpy as np 
import matplotlib.pyplot as plt 

L = 500 #仿真长度

# e(k) = c/d * f(k) f(k)为白色噪声
d = [1, -1.5, 0.7, 0.1] #有色噪声的c
c = [1, 0.5, 0.2]   # 有色噪声的d
nd = len(d) - 1 #  d的阶次
print("nd :", nd)
nc = len(c) - 1  # c的阶次
print("nc :", nc)
Xwn = np.zeros(nc)  # 白色噪声初始值
print("the first of Xwn:", Xwn)
Xcn = np.zeros(nd)  # 有色初始值
print("the first of Xcn:", Xcn)
Xw = np.random.normal(0, 1, L) ##产生均值为0, 方差为1的高斯（正态）分布

Xc = []
for i in range(2):
    Xc1 = map(lambda a, b, c, d: -a * b + c * d, d[1:nd + 1], Xcn, c, [Xw[i], Xwn]) # Xc1为迭代器
    #Xc2 = map(lambda a, b: -a * b, d[1:nd + 1], Xcn)
    list_Xc1 = list(Xc1)  # iter list成数组
    Xc.append(list_Xc1)


print("the res:", Xc)


    

                            
