# -*- coding: utf-8 -*-
"""
Project:
@author: Gene
@Email:GeneWithyou@gmail.com
@Github:github.com/Gene20/DeepLearning
@personalWeb:www.gene20.top
"""
from numpy import *

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]    
#1.相似度计算
#欧氏距离
def ecludSim(inA,inB):
    dist=1.0/(1.0+linalg.norm(inA-inB))
    return dist
#皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA)<3:return 1.0
    pCoef=corrcoef(inA,inB,rowvar=0)[0][1]
    return 0.5+0.5*pCoef
#余弦相似度
def cosSim(inA,inB):
    divide=float(inA.T*inB)
    divisor=linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(divide/divisor)

#2.基于物品相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
        
def recommend(dataMat,user,N=3,simMeth=cosSim,estMeth=standEst):
    notScoItem=nonzero(dataMat[user,:].A==0)[1]
    if len(notScoItem)==0:return '您已经对所有物品进行了评分!'
    itemScores=[]
    for i in notScoItem:
        currScore=estMeth(dataMat,user,simMeth,i)
        itemScores.append((i,currScore))
    return sorted(itemScores,key=lambda pp:pp[1],reverse=True)[:N]

#3.基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


#4.图片压缩函数
def printMat(dataMat,thresh):
    m,n=shape(dataMat)
    for i in range(m):
        for j in range(n):
            if float(dataMat[i,j])>thresh:
                print 1,
            else:
                print 0,
        print ''
        
def imgCompress(numSV=3,thresh=0.8):
    my=[]
    for line in open('0_5.txt').readlines():
        currLine=[]
        for num in line.strip():
            currLine.append(int(num))
        my.append(currLine)
    dataMat=mat(my)
    print '**********未压缩的图片************'        
    printMat(dataMat,thresh)
    U,Sigma,Vt=linalg.svd(dataMat)
    SigSv=mat(eye(numSV)*Sigma[:numSV])
    reMat=U[:,:numSV]*SigSv*Vt[:numSV,:]
    print '**********压缩后重构的图片************'        
    printMat(reMat,thresh)
imgCompress()    
        
        
        
