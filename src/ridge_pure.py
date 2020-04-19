#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from src.solve_sym import solve_sym


def ridge_pure(xtx, xty, x_mean, y_mean, n, p, lam, is_scale=True):
    # lam为list
    xtx_scale = xtx - np.outer(x_mean, x_mean) * n  # 中心化
    xty_scale = xty - x_mean * y_mean * n
    if is_scale:  # 标准化
        x_std = np.sqrt(np.diag(xtx_scale) / (n - 1))
        x_std_mat = 1 / np.repeat(x_std.reshape((1, p)), p, axis=0)
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat
        xty_scale = xty_scale / x_std
    b1 = np.array([
        solve_sym(xtx_scale + np.identity(p) * lam_, xty_scale) for lam_ in lam
    ])
    if is_scale:
        b1 = b1 / x_std  # 原始X对应的系数beta
    b0 = (y_mean - np.dot(b1, x_mean)).reshape(
        (-1, 1))  # beta0 = y_bar - x_bar*beta
    return np.c_[b0, b1]

