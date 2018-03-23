"""
Created on Fri Mar 23 11:47:40 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

import numpy as np
import pylab as pl
import neurolab as nl
import cv2
import mlpy

print("working...")

def readpic(fn):
    #返回图像特征码
    fnimg = cv2.imread(fn)
    img = cv2.resize(fnimg, (500,400))
    w = img.shape[1]
    h = img.shape[0]
    w_interval = w/20
    h_interval = h/10
    alltz = []
    for now_h in range(0, h, h_interval):
        for now_w in range(0, w, w_interval):
            b = img[now_h:now_h+h_interval, now_w:now_w+w_interval, 0]
            g = img[now_h:now_h+h_interval, now_w:now_w+w_interval, 1]
            r = img[now_h:now_h+h_interval, now_w:now_w+w_interval, 2]
            btz = np.mean(b)
            gtz = np.mean(g)
            rtz = np.mean(r)
            alltz.append([btz, gtz, rtz])
    #print(alltz)
    #print(np.array(alltz))
    result_alltz = np.array(alltz).T
    #print(result_alltz)
    pca = mlpy.PCA()
    pca.learn(result_alltz)
    result_alltz = pca.transform(result_alltz, k=len(result_alltz)/2)
    result_alltz = result_alltz.reshape(len(result_alltz))
    return result_alltz

#获得结果函数
def getresult(simjq):
    jq=[]
    for j in range(0, len(simjq)):
        maxjq = -2
        nowii = 0
        for i in range(0, len(simjq[0])):
            if simjq[j][i]>maxjq:
                maxjq=simjq[j][i]
                nowii = i
        jq.append(len(simjq[0]) - nowii)
    return jq

#x和d样本初始化
train_x = []
d = []
sp_d = []
sp_d.append([0,0,1])
sp_d.append([0,1,0])
sp_d.append([1,0,0])

#读取图像
for ii in range(1, 4):
    for jj in range(1, 4):
        fn = 'p' + str(ii) + '-' + str(jj) + '.png'
        pictz = readpic(fn)
        #print(pictz)  #仅为1*3的数组
        train_x.append(pictz)
        d.append(sp_d[ii-1])

myinput = np.array(train_x)
mytarget = np.array(d)
print(train_x)
print(d)
mymax = np.max(myinput)
# print(mymax)
netminmax = []
for i in range(0, len(myinput[0])):
    netminmax.append([0, mymax])
# print(netminmax)
    
print("\nstart to establish network...")
bpnet = nl.net.newff(netminmax, [5,3])
print("\ntraining...")
err = bpnet.train(myinput, mytarget, epochs=800, show=5, goal=0.2)
print(err)
if err[len(err)-1]>0.4:
    print("failed...\n")
else:
    print("\ndone")
    pl.subplot(111)
    pl.plot(err)
    pl.xlabel('Epoch number')
    pl.ylabel('error (default SSE)')
    print("test samples")
    simd = bpnet.sim(myinput)
    mysimd = getresult(simd)
    print(mysimd)
    print("simulating")
    
    testpictz = np.array([readpic('ptest3.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset3.png=====")
    print(simtest)
    print(mysimtest)
    testpictz = np.array([readpic('ptest1.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset1.png=====")
    print(simtest)
    print(mysimtest)
    testpictz = np.array([readpic('ptest2.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset2.png=====")
    print(simtest)
    print(mysimtest)
    testpictz = np.array([readpic('ptest21.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset21.png=====")
    print(simtest)
    print(mysimtest)
    testpictz = np.array([readpic('ptest22.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset22.png=====")
    print(simtest)
    print(mysimtest)
    testpictz = np.array([readpic('ptest31.png')])
    simtest = bpnet.sim(testpictz)
    mysimtest = getresult(simtest)
    
    print("=====ptset31.png=====")
    print(simtest)
    print(mysimtest)
    
    pl.show()