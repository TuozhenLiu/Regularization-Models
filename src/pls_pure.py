import numpy as np
from numpy import linalg as nl
from src.qf import qf


def pls_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, is_var_exp=False, yty=None):
    xtx_scale = xtx - np.outer(x_mean, x_mean)*n # 中心化
    xty_scale = xty - x_mean * y_mean * n
    x_std = None
    if is_scale: # 标准化
        x_std = np.sqrt(np.diag(xtx_scale) / (n - 1))
        x_std_mat = 1 / np.repeat(x_std.reshape((1, p)), p, axis=0) # W^-1
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat
        xty_scale = xty_scale / x_std
    ete = np.copy(xtx_scale)
    etf = np.copy(xty_scale)
    wp_mat = np.eye(p)
    b_ps = []
    if is_var_exp:
        var_y = yty - n * y_mean * y_mean
        var_exp_x = np.zeros(p, dtype=float)
        var_exp_y = np.zeros(p, dtype=float)

    for i in np.arange(p):
        w = etf
        etf_norm = nl.norm(etf) # 模
        w = w / etf_norm # W = ETF / ||ETF||
        t_norm = qf(ete, w) # 二次型， ||t||^2 = WT ETE W
        r = etf_norm / t_norm # y对t 回归系数 
        ws = r * np.dot(wp_mat, w) # wstar y 关于 E 的回归系数
        b_ps.append(np.copy(ws)) # list
        pp = np.dot(ete, w) / t_norm # E被t解释 的 回归系数
        if is_var_exp:
            var_exp_x[i] = np.sum(pp * pp) * t_norm
            var_exp_y[i] = r * r * t_norm 
        if i < (p - 1):
            wp = np.eye(p) - np.outer(w, pp) # I-wpT
            wpt = wp.T
            ete = np.dot(wpt, np.dot(ete, wp))
            etf = np.dot(wpt, etf)
            wp_mat = np.dot(wp_mat, wp)
            
    b_ps = np.array(b_ps)
    b_ps = np.cumsum(b_ps, axis=0)
    if is_scale:
        b_ps = b_ps / x_std
    b0 = y_mean - np.dot(b_ps, x_mean)
    b_ps = np.c_[b0.reshape((p, 1)), b_ps]
    
    if is_var_exp:
        var_exp_x = np.cumsum(var_exp_x)
        var_exp_x = var_exp_x/var_exp_x[-1]
        var_exp_y = np.cumsum(var_exp_y)
        var_exp_y = var_exp_y / var_y
        return b_ps, var_exp_x, var_exp_y
    else:
        return b_ps
