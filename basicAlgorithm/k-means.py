# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:23:40 2018

@author: Administrator
"""
"""
@code: 1051739153@qq.com
"""
 # 加载数据
def loadDataSet(filename,delim='\t'):  
    fr =open(filename)  
    stringArr =[line.strip().split(delim) for line in fr.readlines()]  #这个函数以字符串读入
    datArr =[list(map(float,line)) for line in stringArr]     #这个函数转化为float类型的数字
    return mat(datArr)

def kMeans(dataSet, k):
    m = shape(dataSet)[0]
    n = shape(dataSet)[1]
    clusterAssment = mat(zeros((m,2)))
    centroids = mat(zeros((k,n)))    # 用于记录质心点的坐标
    for index in range(n):
        centroids[:,index] = mat(5+5*random.rand(k,1))
    clusterChanged = True
    while clusterChanged:
        print(centroids)
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                vecA = array(centroids)[j,:]
                vecB = array(dataSet)[i,:]
                distJI = sqrt(sum(power(vecA - vecB, 2)))
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(array(clusterAssment)[:,0]==cent)[0]]
            print(ptsInClust)
            print(nonzero(array(clusterAssment)[:,0]==cent)[0])
            centroids[cent,:] = mean(ptsInClust, axis=0)
    ide = nonzero(array(clusterAssment)[:,0]==cent)[0]
    return centroids, clusterAssment, ide

if __name__=='__main__': 
    # 加载数据  
    dataMat = loadDataSet('testSet.txt')  
    # 
    centroids, clusterAssment, ide = kMeans(dataMat,4)
    # 显示结果  
    fig = plt.figure()  
    ax = fig.add_subplot(111)  
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)  
    ax.scatter(centroids[:,0].flatten().A[0],centroids[:,1].flatten().A[0],marker='o',s=50,c='red')  
    plt.show()  
    