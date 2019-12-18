import os
import numpy as np 
import matplotlib.pyplot as plt 

#the least_square
a = np.array([1, -1.5, 0.7]); b = np.array([1, 0.5]); d = 3 #

len_a = len(a) -1; len_b = len(b) - 1 ; L =500
Input_init = np.zeros(d + len_b) #the init_value of input
Output_init = np.zeros(len_a) # the init_value of ouput

x1 = 1; x2 = 1; x3 = 1; x4 = 0; S = 1 # the init_value of move_register and square_wave

white_noise = np.random.normal(0, 1, L) # the white_noise

phi = np.array([]) # the 
y = np.array([])
U = np.array([]) #the inverse_M_sequence
theta = np.append(a[1: len_a + 1], b) #the permeter of object
for i in range(L):
    # Y = AX + xi(noise)
    temp = np.append(-Output_init, Input_init[d - 1: d + len_b]) # X
    phi = np.concatenate((phi, temp)) 
    #print("the phi: ", phi)

    temp_y = np.dot(temp, theta.transpose()) + white_noise[i] # Y
    #save the y
    y = np.append(y, temp_y) 

    #the inverse_M_sequence
    IM = S ^ x4  
    if IM == 0:
        U = -1
    else:
        U = 1
    S = not(S); M = x3 ^ x4

    #update the data
    x4 = x3; x3 = x2; x2 = x1; x1 = M 

    for k in range(d + len_b - 1, 0, -1):
        Input_init[k] = Input_init[k - 1]
    Input_init[0] = U

    for j in range(len_a - 1, 0, -1):
        Output_init[j] = Output_init[j - 1]
    Output_init[0] = temp_y

# the np.narray to np.matrix
y_matrix = np.matrix(y.reshape(len(y), 1))

phi_matrix = np.matrix(phi.reshape((L, 4)))


print("the phi_matrix_transpose*phi_matrix:", np.matmul(phi_matrix.transpose(), phi_matrix))


temp_phi_phi = np.linalg.inv(np.matmul(phi_matrix.transpose(), phi_matrix))
temp_phi_phi_phi = np.dot(temp_phi_phi, phi_matrix.transpose())

theta_estimate = np.dot(temp_phi_phi_phi, y)

print("the theta_estimate: ", theta_estimate)


