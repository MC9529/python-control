import os
import numpy as np 
import matplotlib.pyplot as plt 

L = 500 # the len of noise

# e(k) = c/d * f(k) f(k) is white_noise  and  e(k) is colored_noise
d = np.array([1, -1.5, 0.7, 0.1]) # the c
c = np.array([1, 0.5, 0.2])   # the d

nd = len(d) - 1 #  
nc = len(c) - 1  # 

white_noise_init = np.zeros(nc)  # the init_value of white_noise

colored_noise_init = np.zeros(nd)  # the init_valuw of colored_noise

white_noise = np.random.normal(0, 1, L) ## the normal distribution

colored_noise = []   # colored_noise
random_walk = [0.0]
for i in range(L):

    old_total_noise = random_walk[i]
    new_noise = white_noise[i]
    random_walk.append(old_total_noise + new_noise)

    noise_1 = np.dot(-d[1 : nd + 1], colored_noise_init.transpose()) ##d * colored_noise_init
    white_noise_init_array = np.insert(white_noise_init, 0, white_noise[i])
    noise_2 = np.dot(white_noise_init_array, c.transpose()) # c * white_noise

    colored_noise.append(noise_1 + noise_2) ## save the colored_noise
    #update the data
    for j in range(nd - 1, 0, -1):
        colored_noise_init[j] = colored_noise_init[j - 1]

    colored_noise_init[0] = noise_2 + noise_1

    for k in range(nc-1, 0, -1):
        white_noise_init[k] = white_noise_init[k - 1]

    white_noise_init[0] = white_noise[i]


random_walk = np.array(random_walk)

#figure
plt.figure(1)
ax = plt.subplot(2, 1, 1)
ax1 = plt.subplot(2, 1, 2)

plt.sca(ax)
ax.plot(white_noise, 'Red', label = 'white_noise')
plt.title('white_noise')
#plt.xlabel("k")
plt.ylabel("amplitude")
ax.legend()

plt.sca(ax1)
ax1.plot(colored_noise, 'Blue', label = 'colored_noise')
plt.title('colored_noise')
plt.xlabel("k")
plt.ylabel("amplitude")
plt.legend()

plt.figure(2)

ax3 = plt.subplot(1, 1, 1)
ax3.plot(random_walk, 'Green' ,label = "random_walk")
plt.title('random_walk')
plt.xlabel("k")
plt.ylabel("amplitude")
plt.legend()

plt.show()

exit()


    

                            
