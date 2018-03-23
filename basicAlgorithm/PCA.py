# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:13:35 2018

@author: wangange
"""

"""
@code: 1051739153@qq.com
"""

from numpy import *
import matplotlib.pyplot as plt 
    
# 加载数据，返回mat型数据  
def loadDataSet(filename,delim='\t'):  
    fr =open(filename)  
    stringArr =[line.strip().split(delim) for line in fr.readlines()]  #这个函数以字符串读入
    datArr =[list(map(float,line)) for line in stringArr]     #这个函数转化为float类型的数字
    return mat(datArr)

def pca(dataMat, dimen):
    meanVals = mean(dataMat, axis=0)       #1
    meanRemoved = dataMat - meanVals       #2
    covMat = cov(meanRemoved, rowvar=0)    #3
    Vals, Vects = linalg.eig(mat(covMat))  #4
    print(Vals)
    print(Vects)
    ValInt = argsort(Vals)                 #5
    print(ValInt)
    ValInd = ValInt[:(-dimen+1):-1]        #6
    redEigVects = Vects[:,ValInd]          #7
    print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects  #8
    reconMat =(lowDDataMat * redEigVects.T) +meanVals  #9 
    return lowDDataMat,reconMat            #10
    
if __name__=='__main__': 
    # 加载数据  
    dataMat = loadDataSet('testSet.txt')  
    # 进行pca，这里将数据转换为1维  
    lowDat,reconMat = pca(dataMat,1)  
    # 显示结果  
    fig = plt.figure()  
    ax = fig.add_subplot(111)  
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)  
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')  
    plt.show()  