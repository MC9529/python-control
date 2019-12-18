import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.linalg import pinv
from scipy.linalg import hankel
from scipy.linalg import svd
from scipy.sparse import triu
from control import tf, ss, mixsyn, step_response
#define the hankel matrix
def hank(g, k):
    H = hankel(g, [1: len(g)/2, (1+k): len(g)/2 + k])
    return H


def ERAOKIDMPC(u, y, r, Nc, Np, lb, ub):
    g = y * pinv(triu(toeplitz(u)))

    H0 = hank(g, 0)
    H1 = hank(g, 1)
    try:
        [U, S, V] = 
        
        