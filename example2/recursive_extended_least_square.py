import os
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.matlib 
#the least_square
a = np.array([1, -1.5, 0.7]); b = np.array([1, 0.5]); c = np.array([1, -1, 0.2]);d = 3 #

len_a = len(a) -1; len_b = len(b) - 1 ; len_c = len(c) - 1; L = 1500

Input_init = np.zeros(d + len_b) #the init_value of input
Output_init = np.zeros(len_a) # the init_value of ouput

white_noise_init = np.zeros(len_c)
white_noise_extimate_init = np.zeros(len_c)

white_noise = np.random.normal(0, 1, L) # the white_noise
white_noise_1 = np.sqrt(0.1) * np.random.normal(0, 1, L)

theta_temp = np.append(a[1: len_a + 1], b)
theta = np.append(theta_temp, c[1: len_c + 1]) #the permeter of object

theta_1 = np.zeros(len_a + len_b + 1 + len_c)
P = np.eye(len_a + len_b + 1 + len_c, len_a + len_b + 1 + len_c)

forgetting_factor = 1 #the forgetting_factor
phi = np.array([]) # the 
thetae_1 = []
thetae_2 = []
thetae_3 = []
thetae_4 = []
thetae_5 = []
thetae_6 = []


for i in range(L):

    temp_temp = np.append(-Output_init, Input_init[d - 1: d + len_b])
    temp = np.append(temp_temp, white_noise_init) # X

    temp_y = np.dot(temp, theta.transpose()) + white_noise_1[i] # Y
    
    temp_estimate_temp = np.append(-Output_init, Input_init[d - 1: d + len_b])
    temp_estimate = np.append(temp_estimate_temp, white_noise_extimate_init)


    temp_estimate_matrix = np.matrix(temp_estimate)
    P_matrix = np.matrix(P)

    K = (P_matrix * temp_estimate_matrix.transpose()) /(forgetting_factor + temp_estimate_matrix * P_matrix * temp_estimate_matrix.transpose())
    thetae_temp = theta_1 + K * (temp_y - temp_estimate * theta_1)
    
    #print("the size of thetae: ", thetae_temp.shape)
    thetae_1.append(thetae_temp[0,0])
    thetae_2.append(thetae_temp[1,1])
    thetae_3.append(thetae_temp[2,2])
    thetae_4.append(thetae_temp[3,3])
    thetae_5.append(thetae_temp[4,4])
    thetae_6.append(thetae_temp[5,5])

    P = (np.eye(len_a + len_b + len_c + 1) - K * temp_estimate_matrix) * P /forgetting_factor
    ##the estimate of white
    white_noise_estimate = temp_y - temp_estimate * thetae_temp 

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
ax.plot(thetae_5, 'brown', label = 'thetae_5')
ax.plot(thetae_6, 'cyan', label = 'thetae_6')
#ax.plot(theta_real_3, 'burlywood', label = 'theta_real_3', linestyle="-" )
#ax.plot(theta_real_4, 'coral', label = 'theta_real_4', linestyle="-" )
plt.title('thetae')
plt.xlabel("k")
plt.ylabel("value")

ax.legend()
plt.show()
