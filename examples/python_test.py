import os
import numpy as np 
import matplotlib.pyplot as plt 

e = np.array([1, 2, 3])
f = np.array([2, 3, 4])
a = np.append(e[1:3], f,axis = 0)
m = np.array([[1,2,3], [2,2,3], [2,3,4]])
print("the inverse:", type(np.linalg.inv(m)))
g = np.vdot(e, f.transpose())
print("the g: ", g)

print("the a: ", a)

for i in range(len(e) -1, 0, -1):
              e[i] = e[i - 1]
              print("the e:", e)

#e = np.delete(e, 0)
#e = np.insert(e, 0 ,5)
e[0] = 5
print("the e:" , e)


f = np.insert(f, 0, 5)
print("the f :", f)

f_list = list(f)


f_list.insert(0, 1)
print("the f:", f_list)

f_array = np.asarray(f_list)

print("the f:", f)

#g = np.dot(e, f.transpose())
    
#print("the c:", g)