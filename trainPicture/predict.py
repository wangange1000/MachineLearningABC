# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:15:27 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tensorflow.python.lib.io import file_io
import os
import sys
import numpy as np
import pickle
import scipy
from tflearn.datasets import cifar10

def load_data(dirname="cifar-10-batches-py", one_hot=False):
    X_train=[]
    Y_train=[]
    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i==1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)
    
    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)
    
    #print(X_train) #60000张图片形成60000X3072的矩阵
    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))/255.
    #print(X_train) #形成60000X1024X3的矩阵
    X_train = np.reshape(X_train, [-1,32,32,3])
    #print(X_train)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))/255.
    X_test = np.reshape(X_test, [-1,32,32,3])
    
    if one_hot:
        Y_train = to_categorical(Y_train, 10)
        Y_test = to_categorical(Y_test, 10)
    
    return (X_train, Y_train), (X_test, Y_test)

def load_batch(fpath):
    objectFile = file_io.read_file_to_string(fpath, 'rb') #！注意：这个地方一定要加上‘rb’，否则编码错误不运行
    #origin_bytes = bytes(object, encoding='latin1')
    #with open(fpath, 'rb') as f:
    if sys.version_info > (3,0):
        d = pickle.loads(objectFile, encoding='latin1')
    else:
        d = pickle.loads(objectFile)
        
    data = d["data"]
    labels =  d["labels"]
    return data, labels

(X,Y), (X_test, Y_test) = load_data()
X, Y = shuffle(X,Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)
tf.reset_default_graph()

#实时数据预处理
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

#构建卷积网络
network = input_data(shape=[None, 32,32,3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0)

#导入模型
model.load("classifier.tfl")
#导入预测集
img = scipy.ndimage.imread('test_fly.jpg', mode="RGB")
img = scipy.misc.imresize(img, (32,32), interp="bicubic").astype(np.float32, casting='unsafe')
#输出预测结果
prediction = model.predict([img])
print(prediction[0])

img = scipy.ndimage.imread('horse_test.jpg', mode="RGB")
img = scipy.misc.imresize(img, (32,32), interp="bicubic").astype(np.float32, casting='unsafe')
prediction = model.predict([img])
print(prediction[0])

img = scipy.ndimage.imread('test_ship.jpg', mode="RGB")
img = scipy.misc.imresize(img, (32,32), interp="bicubic").astype(np.float32, casting='unsafe')
prediction = model.predict([img])
print(prediction[0])