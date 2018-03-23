# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:48:00 2018

@author: wangange
"""
"""
@code: 1051739153@qq.com
"""

import math
import operator

def euclideanDistance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += pow((inst1[x] - inst2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

if __name__=='__main__': 
    trainSet=[[1,1,1,'a'],[2,2,2,'a'],[1,1,3,'a'],[4,4,4,'b'],[0,0,0,'a'],[4,4.5,4,'b']]
    testInstance = [5,5,5]
    k=5
    neighbors = getNeighbors(trainSet, testInstance, k)
    response = getResponse(neighbors)
    print("\nNeighbors are: ")
    print(neighbors)
    print("\nResponse is: ")
    print(repr(response))