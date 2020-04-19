#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from src.ridge_pure import ridge_pure


class Ridge(object):
    def __init__(self, x, y, x_names=None, lam=0, is_scale=True):
        self.n, self.p = np.shape(x)
        self.x = x
        self.y = y
        self.xtx = np.dot(x.T, x)
        self.xty = np.dot(x.T, y)
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y)
        self.b = 0
        self.lam = lam
        self.is_scale = is_scale
        self.cv_err = 0
        if x_names == None:
            self.x_names = list(range(1, self.p + 1))
        else:
            self.x_names = x_names

    def ridge(self):  #计算找到对应的lambda后，输出系数
        self.b = ridge_pure(self.xtx, self.xty, self.x_mean, self.y_mean,
                            self.n, self.p, self.lam, self.is_scale)

    def cv(self, k=10):
        indexs = np.array_split(np.random.permutation(np.arange(0, self.n)), k)

        def cvk(index):
            tx = self.x[index]
            tn, tp = np.shape(tx)
            if tn == 1:
                tx = tx.reshape((1, self.p))  # 防止k=1的时候后面矩阵运算出错
            tn_ = self.n - tn
            ty = self.y[index]
            txt = tx.T
            txx_ = self.xtx - np.dot(txt, tx)
            txy_ = self.xty - np.dot(txt, ty)
            tx_sum = np.sum(tx, axis=0)
            ty_sum = np.sum(ty)
            tx_mean_ = (self.n * self.x_mean - tx_sum) / tn_
            ty_mean_ = (self.n * self.y_mean - ty_sum) / tn_
            tb = ridge_pure(txx_, txy_, tx_mean_, ty_mean_, tn_, tp, self.lam,
                            self.is_scale)
            tx = np.c_[np.ones((tn, 1)), tx]
            ty_pred = np.dot(tb, tx.T)  # tb lxp  tx.T pxn
            err = ty_pred - ty  # lxn 每行一个模型,一共l个（lam的数量）
            err = err * err
            return np.sum(err, axis=1)  # 每行（每个模型）取总和

        #self.cv_err = [[cvk(index, lambd) for lambd in self.lam] for index in indexs]
        self.cv_err = np.sum(np.array([cvk(index) for index in indexs]),
                             axis=0) / self.n
        min_k = np.argmin(self.cv_err)
        self.cv_b = self.b[min_k]
        print('best lambda:', self.lam[min_k])
        return self.cv_b  # 返回cv下最优的系数

    def report_coe(self):
        names = np.append("inter", self.x_names)
        results = pd.DataFrame(self.b,
                               columns=names,
                               index=np.arange(1,
                                               len(self.lam) + 1))
        results["cverr"] = self.cv_err
        results["lam"] = self.lam
        return results

    def predict(self, xn):  # 给测试集x，预测y
        tn, _ = np.shape(xn)
        xn_ = np.c_[np.ones((tn, 1)), xn]
        return np.dot(self.cv_b, xn_.T)  # 用 cv 中最小 err 对应的 cv_b 作为系数beta

    def predict_err(self, xn, yn):  # 给测试集xy，得到cv下err
        err = yn - self.predict(xn)
        err = err * err
        return np.mean(err)

    def test_err(self, xn, yn):  # 给测试集xy，得到外样本下err
        tn, _ = np.shape(xn)
        xn_ = np.c_[np.ones((tn, 1)), xn]
        err = yn - np.dot(
            self.b,
            xn_.T,
        )
        # p个模型都算一次，比较外样本测试误差。
        # cv 中最小 err 对应的模型 外样本预测不一定最小
        err = err * err
        err_mean = np.mean(err, axis=1)
        err_std = np.std(err, axis=1, ddof=1) / np.sqrt(tn)
        result = {'err_mean': err_mean, 'err_std': err_std, 'lam': self.lam}
        print('best lambda:', self.lam[np.argmin(err_mean)])
        print('err:', np.min(err_mean))
        return pd.DataFrame(result, index=np.arange(1, len(self.lam) + 1))

