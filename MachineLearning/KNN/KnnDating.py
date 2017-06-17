# -*- coding: utf-8 -*-
"""
Project:KNN算法
@author: Gene
@Email:GeneWithyou@gmail.com
@Github:github.com/Gene20/DeepLearning
@personalWeb:www.gene20.top
"""

from numpy import *
import operator
from os import listdir

#1.使用k-临近算法将每组数据划分到某个类中
#1.Each group of data is divided into a class by using the k- algorithm
#输入：inX 分类的向量 / dataSet 训练样本 / labels标签向量 / k 最邻近的个数
#输出：sortedClassCount[0][0] 频率最高的元素标签
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
    
    
#2.将文本记录转换成NumPy的解析程序
#2.Parsing a text record into a NumPy    
#返回  returnMat：Nx3的特征向量矩阵   classLabelVector: Nx1的分类标签矩阵
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
#datingDataMat,datingLabels=file2matrix('datingTestSet2.txt') 

#3.归一化特征值（公式: newValue=(oldValue-min)/(max-min)）
#3.Normalized eigenvalue(formula：newValue=(oldValue-min)/(max-min))  
#返回：normDataSet 特征值在0-1区间的新矩阵   ranges 取值范围   minVals 每列的最小值变量
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
    
#normDataSet,ranges,minVals=autoNorm(datingDataMat)

#4.分类器针对约会网站的测试
#4.Classifier for dating site testing
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "分类器返回的类型是: %d, 真正的类型是: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "总的错误率为: %f" % (errorCount/float(numTestVecs))
    print errorCount
    return errorCount/float(numTestVecs)

errorDate=datingClassTest()
print("使用KNN算法针对约会网站的测试准确率为: %f%%"%((1-errorDate)*100))


#5.具体实现用户输入数据进行预测(完整的预测系统)
#5.Specific implementation of user input data to predict(Complete forecasting system)
def classifyPerson():
    resultList=['一点都不适合','可以考虑','跟你很配，建议去见面']
    percentGames=float(raw_input("玩视频游戏所耗时间百分比："))
    flyMiles=float(raw_input("每年获得的飞行常客里程数："))
    eatIceCream=float(raw_input("每周消费的冰淇淋公升数："))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([flyMiles,percentGames,eatIceCream])
    theResult=classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "使用KNN算法预测该男士是否适合约会？",resultList[theResult-1]
    
#classifyPerson()    
      
