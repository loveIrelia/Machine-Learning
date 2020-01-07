import numpy as np
def caculator(u,q,TestVec,k):
    Result = (-(TestVec-u)**2/(2*q**2))+np.log(1/(np.sqrt(2*3.1415926)*q))+np.log(1/k)
    return Result