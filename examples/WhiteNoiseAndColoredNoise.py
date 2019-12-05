import os
import numpy as np 
import matplotlib.pyplot as plt 

L = 500 #仿真长度

# e(k) = c/d * f(k) f(k)为白色噪声
d = np.array([1, -1.5, 0.7, 0.1]) #有色噪声的c
c = np.array([1, 0.5, 0.2])   # 有色噪声的d

nd = len(d) - 1 #  d的阶次
nc = len(c) - 1  # c的阶次

white_noise_init = np.zeros(nc)  # 白色噪声初始值

colored_noise_init = np.zeros(nd)  # 有色初始值

white_noise = np.random.normal(0, 1, L) ##产生均值为0, 方差为1的高斯（正态）分布 白噪声

colored_noise = []   # 有色噪声
random_walk = [0.0]
for i in range(L):

    old_total_noise = random_walk[i]
    new_noise = white_noise[i]
    random_walk.append(old_total_noise + new_noise)

    noise_1 = np.dot(-d[1 : nd + 1], colored_noise_init.transpose()) ##d * colored_noise_init
    white_noise_init_array = np.insert(white_noise_init, 0, white_noise[i])
    noise_2 = np.dot(white_noise_init_array, c.transpose()) # c * white_nois

    colored_noise.append(noise_1 + noise_2) ##有色噪声保存
    #数据更新
    for j in range(nd - 1, 0, -1):
        colored_noise_init[j] = colored_noise_init[j - 1]

    colored_noise_init[0] = noise_2 + noise_1

    for k in range(nc-1, 0, -1):
        white_noise_init[k] = white_noise_init[k - 1]

    white_noise_init[0] = white_noise[i]


random_walk = np.array(random_walk)

#画图
plt.figure(1)
ax = plt.subplot(2, 1, 1)
ax1 = plt.subplot(2, 1, 2)

plt.sca(ax)
ax.plot(white_noise, 'Red', label = 'white_noise')
plt.title('white_noise')
ax.legend()

plt.sca(ax1)
ax1.plot(colored_noise, 'Blue', label = 'colored_noise')
plt.title('colored_noise')
plt.legend()

plt.figure(2)

ax3 = plt.subplot(1, 1, 1)
ax3.plot(random_walk, 'Green' ,label = "random_walk")
plt.title('random_walk')
plt.legend()

plt.show()

exit()


    

                            
