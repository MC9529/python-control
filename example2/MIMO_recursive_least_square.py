import os
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.matlib 

#the least_square
a = np.array([1, -0.8, -0.2, 0.6])
b = np.array([
             [[3, -3.5, -1.5, 0.0], 
             [1, -0.2, -0.5, 0]],
             [[0, -4, -2, -1],
             [1, -1.5, 0.5, 0.2]]
             ])

d = np.array([[1, 1],[2, 1]])
nb = np.array([[2, 2],[2, 3]])

len_a = len(a) -1; len_b = len(b) ; r = 2; m = 2; L =400
mb = np.max(d + nb)  # d + nb的最大元素

Input_init = np.zeros((r, mb)) #the init_value of input
Output_init = np.zeros((len_a, m)) # the init_value of ouput
white_noise_init = np.zeros((len_a, m)) 


white_noise = np.array([np.sqrt(0.1)*np.random.normal(0, 1, L), 
                        np.sqrt(0.1)*np.random.normal(0, 1, L)]) # the white_noise
print("the white_noise:", white_noise)
white_noise_1 = np.array([np.random.normal(0, 1, L),np.random.normal(0, 1, L)])

theta1 = []; theta2 = []
for i in range(r):
    theta1 = np.append(theta1, b[0][i][d[0][i] - 1: d[0][i] + nb[0][i]] )
    theta2 = np.append(theta2, b[1][i][d[1][i] - 1: d[1][i] + nb[1][i]] )
    

theta_temp = np.append(theta1, theta2)
theta = np.append(a[1: len_a + 1], theta_temp)

theta_init = np.zeros((len_a + sum(sum(nb)) + 4, 1))

#P = np.eye(len_a + len_b + 1, len_a + len_b + 1)
P = np.eye(len_a + sum(sum(nb)) + 4, len_a + sum(sum(nb)) + 4)

phi = np.array([]) # the 
thetae_a1 = []
thetae_a2 = []
thetae_a3 = []

thetae_b1_1 = []
thetae_b1_2 = []
thetae_b1_3 = []

thetae_b2_1 = []
thetae_b2_2 = []
thetae_b2_3 = []

thetae_b3_1 = []
thetae_b3_2 = []
thetae_b3_3 = []

thetae_b4_1 = []
thetae_b4_2 = []
thetae_b4_3 = []
thetae_b4_4 = []


for i in range(L):
    # Y = AX + xi(noise)
    temp_x1_1 = []; temp_x2_1 = []
    for l in range(r):
        temp_x1_1 = np.append(temp_x1_1, Input_init[0][d[0][l] - 1: d[0][l] + nb[0][l]])
        temp_x2_1 = np.append(temp_x2_1, Input_init[1][d[1][l] - 1: d[1][l] + nb[1][l]])

    temp_x1 = np.append(temp_x1_1, np.zeros(len(temp_x1_1)))
    temp_x2 = np.append(temp_x2_1, np.zeros(len(temp_x2_1)))
    temp_x = np.append(temp_x1, temp_x2)
    temp = np.append(-Output_init, temp_x) # X
    temp = temp.reshape(2, 16)

    matrix_temp = np.mat(temp)
    
    e =  np.append(white_noise[:, 1], white_noise_init ).reshape(4, 2).transpose()
    matrix_e = np.mat(e)
    matrix_a = np.mat(a.reshape(4, 1))
    noise = matrix_e * matrix_a
    print("the type of temp:", temp)
    print("the shape of temp: ", np.shape(temp))
    print("the shape of theta: ", np.shape(theta))
    #temp_y = np.dot(temp, theta.transpose()) + noise # Y
    matrix_theta = np.mat(theta)
    print("the shape of matrix_theta: ", np.shape(matrix_theta))

    temp_y = matrix_temp * matrix_theta.transpose() + noise
    #temp_y = np.dot(temp, theta.transpose())
    print("the temp_y:", temp_y)
    #temp_matrix = np.matrix(temp_y)
    P_matrix = np.matrix(P)
   
    matrix_mi = np.matlib.eye(m, m)

    K = (P_matrix * matrix_temp.transpose()) * np.linalg.inv((matrix_mi + matrix_temp * P_matrix * matrix_temp.transpose()))
    thetae_temp = theta_init + K * (temp_y - matrix_temp * theta_init)
    
   
    thetae_a1.append(thetae_temp[0, 0])
    thetae_a2.append(thetae_temp[1, 0])
    thetae_a3.append(thetae_temp[2, 0])

    thetae_b1_1.append(thetae_temp[3, 0])
    thetae_b1_2.append(thetae_temp[4, 0])
    thetae_b1_3.append(thetae_temp[5, 0])

    thetae_b2_1.append(thetae_temp[6, 0])
    thetae_b2_2.append(thetae_temp[7, 0])
    thetae_b2_3.append(thetae_temp[8, 0])

    thetae_b3_1.append(thetae_temp[9, 0])
    thetae_b3_2.append(thetae_temp[10, 0])
    thetae_b3_3.append(thetae_temp[11, 0])

    thetae_b4_1.append(thetae_temp[12, 0])
    thetae_b4_2.append(thetae_temp[13, 0])
    thetae_b4_3.append(thetae_temp[14, 0])
    thetae_b4_4.append(thetae_temp[15, 0])

    P = (np.eye(len_a + sum(sum(nb)) + 4) - K * temp) * P 

    #update the data
    theta_1 = thetae_temp

    for k in range(r - 1, -1, -1):
        for j in range(mb - 1, -1, -1):
            Input_init[k][j] = Input_init[k][j -1]

        Input_init[k][0] = white_noise[k][i]

    print("the Input_init:", Input_init)
    for k in range(m -1, 0 , -1):
        for j in range(len_a - 1, 0, -1):
            Output_init[j][k] = Output_init[j - 1][k]
            white_noise_init[j][k] = white_noise_init[j - 1][k]

        Output_init[k][0] = temp_y[k, :]
        white_noise_init[0][k] = white_noise[k][i]
    print("the onput_init:", Output_init)
    


# the np.narray to np.matrix

plt.figure(1)
ax = plt.subplot(1, 1, 1)

plt.sca(ax)
ax.plot(thetae_a1, 'Red', label = 'thetae_a1')
ax.plot(thetae_a2, 'Blue', label = 'thetae_a2')
ax.plot(thetae_a3, 'Green', label = 'thetae_a3')
#ax.plot(thetae_b1_1, 'Yellow', label = 'thetae_b1_1')

plt.title('thetae')
plt.xlabel("k")
plt.ylabel("value")

ax.legend()
plt.show()

