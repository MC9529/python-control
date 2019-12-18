import os
import numpy as np 
import matplotlib.pyplot as plt 

L = 30 # the len of sequence
x1 = 1; x2 = 1; x3 = 1; x4 = 0 # the init_valce of register
#print("the x4 :" , x4)
S = 1
U = [] # Inverse_M_sequence
M = [] # M_sequence
for i in range(1, L + 1, 1):
    IM = S ^ x4
    if IM == 0:
        U.append(-1)   # Inverse_M_sequence
    else:
        U.append(1)  # Inverse_M_sequence
    
    S = not(S)
    m = x3 ^ x4  
    M.append( m ) # M_sequence
    x4 = x3; x3 = x2; x2 = x1 ; x1 = m

print("the M_sequence:", M)
print("the_inverse_M_sequence:", U)
#figure
plt.figure(1)
ax = plt.subplot(2, 1, 1)
ax1 = plt.subplot(2, 1, 2)

plt.sca(ax)
ax.plot(M, 'Red', label = 'M_sequence')
plt.title('M_sequence')
#plt.xlabel("k")
plt.ylabel("amplitude")
ax.legend()

plt.sca(ax1)
ax1.plot(U, 'Blue', label = 'Inverse_M_sequence')
plt.title('Inverse_M_sequence')
plt.xlabel("k")
plt.ylabel("amplitude")
plt.legend()


plt.show()

exit()




