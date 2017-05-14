# -*- coding: utf-8 -*-
"""
Project:use PCA to simplify data
@author: Gene
@Email:GeneWithyou@gmail.com
@Github:github.com/Gene20/DeepLearning
@personalWeb:www.gene20.top
"""
from numpy import *
#1.PCA算法
def loadDataSet(filename,delim='\t'):
    fr=open(filename)
    currLine=[line.strip().split(delim) for line in fr.readlines()]
    dataArr=[map(float,num) for num in currLine]
    return mat(dataArr)

def pca(dataMat,topNFet=999999):
    meanData=mean(dataMat,axis=0)
    noMeanData=dataMat-meanData
    convMat=cov(noMeanData,rowvar=0)
    fetVal,fetVec=linalg.eig(mat(convMat))
    valIndex=argsort(fetVal)
    vecIndex=valIndex[:-(topNFet+1):-1]
    nFetVec=fetVec[:,vecIndex]
    lowDData=noMeanData*nFetVec
    iniDataMat=lowDData*nFetVec.T+meanData
    return lowDData,iniDataMat
'''
dataMat=loadDataSet('testSet.txt')
lowData,iniData=pca(dataMat,1)

import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(iniData[:,0].flatten().A[0],iniData[:,1].flatten().A[0],marker='o',s=50,c='red')
'''

'''2.实例：利用PCA对半导体制造数据降维'''
#1.键Nan代替成平均值的函数
def replaceNanWithMean():
    dataMat=loadDataSet('secom.data',' ')
    t=shape(dataMat)[1]
    for i in range(t):
        tMean=mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i]=tMean
    return dataMat

dataMat=replaceNanWithMean()
lowDData,iniDataMat=pca(dataMat,3)

import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(lowDData[:,0].flatten().A[0],lowDData[:,1].flatten().A[0],lowDData[:,2].flatten().A[0],marker='o',s=1,c='red')

