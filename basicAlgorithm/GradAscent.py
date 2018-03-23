# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:08:42 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

from numpy import *

def sigmoid(x):
    return 1.0/(1+exp(-x))

def gradAscent(dataMat, classLabel, alpha, maxCycles):
    dataMatrix = mat(dataMat)
    labelMat = mat(classLabel).transpose()
    m,n = shape(dataMatrix)
    weights = ones((n,1))
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

if __name__=='__main__':
    alpha = 0.001
    maxCycles = 500
    