# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:52:26 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

import numpy as np
import matplotlib.pyplot as plt
import mlpy as mp

f = np.loadtxt("spiral.data")
x, y = f[:, :2], f[:, 2]
svm = mp.LibSvm(svm_type='c_svc', kernel_type='rbf', gamma=100)
svm.learn(x, y)
xmin, xmax = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
ymin, ymax = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
xnew = np.c_[xx.ravel(), yy.ravel()]
ynew = svm.pred(xnew).reshape(xx.shape)
fig = plt.figure(1)
plt.pcolormesh(xx, yy, ynew)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()