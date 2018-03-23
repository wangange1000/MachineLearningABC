# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:14:31 2018

@author: Administrator
"""

from libsvm.python.svmutil import *
from libsvm.python.svm import *

y, x = svm_read_problem('trainData//train.1')
yt, xt =  svm_read_problem('trainData//test.1')
m = svm_train(y, x)
print('test:')
p_label, p_acc, p_val = svm_predict(yt, xt, m)
print(p_label)
