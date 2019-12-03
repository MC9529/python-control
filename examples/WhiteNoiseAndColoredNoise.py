import os
import numpy as np 
import matplotlib.pyplot as plt 

L = 500 #仿真长度

# e(k) = c/d * f(k) f(k)为白色噪声
c = [1, -1.5, 0.7, 0.1] #有色噪声的c
d = [1, 0.5, 0.2]   # 有色噪声的d
nd = len(c) - 1 #  d的阶次
print("nd :", nd)
nc = len(d) - 1  # c的阶次
print("nc :", nc)
Xwn = np.zeros(nc)  # 白色噪声初始值
print("the first of Xwn:", Xwn(0))
Xcn = np.zeros(nd)  # 有色初始值
print("the first of Xcn:", Xcn(0))

                            
