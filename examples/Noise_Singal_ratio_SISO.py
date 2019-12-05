import os
import numpy as np 
import matplotlib.pyplot as plt 

a = np.array([1, -0.4]); b = np.array([1]); c = np.array([1, -0.4])
na = len(a) -1; nb = len(b) - 1; nc = len(c) - 1
#print("the na: ", na)

n = max(na, nb, nc)
print("the max of (na, nb, nc):", n)
a0 = a; b0 = b; c0 = c
for i in range(na, n, 1):
    a0 = np.append(a0, 0)

for i in range(nb, n, 1):
    b0 = np.append(b0, 0)

for i in range(nc, n, 1):
    c0 = np.append(c0, 0)

print("the a0: ", b0)

p = []; qg = []; qh = []
deltau2 = 1; deltav2 = 1
for i in range(n):
    p.append(a0[i])
    qg.append(b0[i])
    qh.append(c0[i])

for i in range(n - 1, 0 , -1):
    for j in range(0, 2, 1):
        p_2[j, i] = p[0] * p[i]

