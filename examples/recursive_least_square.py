import os
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.matlib 

#the least_square
a = np.array([1, -1.5, 0.7]); b = np.array([1, 0.5]); d = 3 #

len_a = len(a) -1; len_b = len(b) - 1 ; L =400
Input_init = np.zeros(d + len_b) #the init_value of input
Output_init = np.zeros(len_a) # the init_value of ouput


white_noise = np.random.normal(0, 1, L) # the white_noise
white_noise_1 = np.sqrt(0.1) * np.random.normal(0, 1, L)

theta = np.append(a[1: len_a + 1], b) #the permeter of object
print("the shape of theat:", np.shape(theta))

theta_1 = np.zeros(len_a + len_b + 1)
print("the theta_1:", np.shape(theta_1))
P = np.eye(len_a + len_b + 1, len_a + len_b + 1)

phi = np.array([]) # the 
thetae_1 = []
thetae_2 = []
thetae_3 = []
thetae_4 = []


for i in range(L):
    # Y = AX + xi(noise)
    temp = np.append(-Output_init, Input_init[d - 1: d + len_b]) # X
    temp_y = np.dot(temp, theta.transpose()) + white_noise_1[i] # Y

    phi = np.concatenate((phi, temp)) 
    temp_matrix = np.matrix(temp)
    P_matrix = np.matrix(P)

    K = (P_matrix * temp_matrix.transpose()) /(1 + temp_matrix * P_matrix *temp_matrix.transpose())
    thetae_temp = theta_1 + K * (temp_y - temp * theta_1)
    
    thetae_1.append(thetae_temp[0,0])
    thetae_2.append(thetae_temp[1,1])
    thetae_3.append(thetae_temp[2,2])
    thetae_4.append(thetae_temp[3,3])
    P = (np.eye(len_a + len_b  + 1) - K * temp) * P 

    #update the data
    theta_1 = thetae_temp

    for k in range(d + len_b - 1, 0, -1):
        Input_init[k] = Input_init[k - 1]
    Input_init[0] = white_noise[i]

    for j in range(len_a - 1, 0, -1):
        Output_init[j] = Output_init[j - 1]
    Output_init[0] = temp_y

# the np.narray to np.matrix

plt.figure(1)
ax = plt.subplot(1, 1, 1)

plt.sca(ax)
ax.plot(thetae_1, 'Red', label = 'thetae_1')
ax.plot(thetae_2, 'Blue', label = 'thetae_2')
ax.plot(thetae_3, 'Green', label = 'thetae_3')
ax.plot(thetae_4, 'Yellow', label = 'thetae_4')
plt.title('thetae')
plt.xlabel("k")
plt.ylabel("value")

ax.legend()
plt.show()


