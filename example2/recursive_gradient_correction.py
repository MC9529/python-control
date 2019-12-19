import os
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.matlib 
#the least_square
a = np.array([1, -1.5, 0.7]); b = np.array([1, 0.5]);d = 3 #

len_a = len(a) -1; len_b = len(b) - 1 ; L = 1500

Input_init = np.zeros(d + len_b) #the init_value of input
Output_init = np.zeros(len_a) # the init_value of ouput

white_noise = np.random.normal(0, 1, L) # the white_noise

theta = np.append(a[1: len_a + 1], b)

theta_1 = np.zeros(len_a + len_b + 1)

alpha = 1
c = 0.1

phi = np.array([]) # the 
thetae_1 = []
thetae_2 = []
thetae_3 = []
thetae_4 = []


for i in range(L):

    temp = np.append(-Output_init, Input_init[d - 1: d + len_b])

    temp_y = np.dot(temp, theta.transpose())# Y
    
    thetae_temp = theta_1 + alpha * temp * (temp_y - temp * theta_1)/(np.dot(temp, temp) + c)
    
    #print("the size of thetae: ", thetae_temp.shape)
    thetae_1.append(thetae_temp[0,0])
    thetae_2.append(thetae_temp[1,1])
    thetae_3.append(thetae_temp[2,2])
    thetae_4.append(thetae_temp[3,3])

    #update the data
    theta_1 = thetae_temp

    for k in range(d + len_b - 1, 0, -1):
        Input_init[k] = Input_init[k - 1]
    Input_init[0] = white_noise[i]

    for j in range(len_a - 1, 0, -1):
        Output_init[j] = Output_init[j - 1]
    Output_init[0] = temp_y

    
plt.figure(1)
ax = plt.subplot(1, 1, 1)

plt.sca(ax)
ax.plot(thetae_1, 'Red', label = 'thetae_1')
ax.plot(thetae_2, 'Blue', label = 'thetae_2')
ax.plot(thetae_3, 'Green', label = 'thetae_3')
ax.plot(thetae_4, 'Yellow', label = 'thetae_4')
#ax.plot(theta_real_3, 'burlywood', label = 'theta_real_3', linestyle="-" )
#ax.plot(theta_real_4, 'coral', label = 'theta_real_4', linestyle="-" )
plt.title('thetae')
plt.xlabel("k")
plt.ylabel("value")

ax.legend()
plt.show()
