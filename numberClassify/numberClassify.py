# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:54:56 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

import cv2
import numpy as np
import mlpy
print('loading...')

def getnumc(fn):
    """返回数字特征"""
    fnimg = cv2.imread(fn)
    img = cv2.resize(fnimg, (8,8))
    alltz=[]
    
    for now_h in range(0,8):
        xtz=[]
        for now_w in range(0,8):
            b = img[now_h, now_w, 0]
            g = img[now_h, now_w, 1]
            r = img[now_h, now_w, 2]
            btz = 255 - b
            gtz = 255 - g
            rtz = 255 - r
            if btz>0 or gtz>0 or rtz>0:
                nowtz = 1
            else:
                nowtz = 0
            xtz.append(nowtz)
        alltz+=xtz
    return alltz

#读取样本数字
x=[]
y=[]
for numi in range(1,10):
    for numij in range(1,5):
        fn = 'nums/' + str(numi) + '-' + str(numij) +'.png'
        x.append(getnumc(fn))
        y.append(numi)
        
x = np.array(x)
y = np.array(y)
svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='poly', gamma=10)
svm.learn(x,y)
print(svm.pred(x))

for iii in range(1,10):
    testfn = 'nums/test/' + str(iii) + '-test.png'
    testx = []
    testx.append(getnumc(testfn))
    print(svm.pred(testx))