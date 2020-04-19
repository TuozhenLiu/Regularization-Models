import numpy as np
from scipy import linalg


def pcr_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, is_var_exp=False, yty=None):
    xtx_scale = xtx - np.outer(x_mean, x_mean)*n # 中心化
    xty_scale = xty - x_mean * y_mean * n # 中心化
    x_std = None
    if is_scale:
        x_std = np.sqrt(np.diag(xtx_scale)/(n-1)) # x标准差 p维
        x_std_mat = 1/np.repeat(x_std.reshape((1, p)), p, axis=0) # x标准差的倒数矩阵 pxp维
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat # 标准化后的xTx（相当于用x除以std后再xTx）
        xty_scale = xty_scale/x_std # 标准化后的xTy
    s, v = linalg.eigh(xtx_scale)
    s = s[-1::-1] # lambda
    v = v[:, -1::-1] # P
    vs = v/s 
    xty_scale = np.dot(v.T, xty_scale)  # PTXTY
    b1 = (np.cumsum(vs * xty_scale, axis=1)).T # 标准化X对应的系数beta_x=P*beta_pc=P (1/s) PTXT Y
    if is_scale:
        b1 = b1 / x_std  # 原始X对应的系数beta
    b0 = (y_mean - np.dot(b1, x_mean)).reshape((p, 1)) # beta0 = y_bar - x_bar*beta
    if is_var_exp:
        var_y = yty - n * y_mean * y_mean # SSE=var(y)
        var_exp_x = np.cumsum(s) 
        var_exp_x = var_exp_x / var_exp_x[-1]  # 主成分累计贡献率
        var_exp_y = xty_scale * xty_scale / s  # 每个主成分对回归平方和的贡献 PTXTY * PTXTY /lambda
        var_exp_y = np.cumsum(var_exp_y) / var_y  # R2
        return np.c_[b0, b1], var_exp_x, var_exp_y
    else:
        return np.c_[b0, b1] # pxp 第一行是一个成分的模型，第二行是两个成分的模型。。。
