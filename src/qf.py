import numpy as np

def qf(a_mat, x):
    # 二次型
    return np.sum(x * np.dot(a_mat, x))
