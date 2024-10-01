import numpy as np

"""
v: multidimensional numpy array
return: vector of result
"""
def sum_s_norm(val):
    return np.sum(val, axis=0)

def prod_t_norm(val):
    n = 0
    for _, v in val.items():
        n = max(v.shape[0], n)
    print(n)
    res = np.zeros(n)
    for _, x in val.items():
        res = np.multiply(res, x)
    return res

