# -*- coding: utf-8 -*-
"""
Project:KNN算法
@author: Gene
@Email:GeneWithyou@gmail.com
@Github:github.com/Gene20/DeepLearning
@personalWeb:www.gene20.top
"""

#手写数字图片:训练样本 2034 测试样本 996
#文件夹testDigits里面是测试样本,这里做一个小演示，就只用了996的样本,注意文件名第一位数字是这个文件的类别
#文件夹trainingDigits里面是训练样本，为2034个特征向量,你也可以再加入过多的样本进行训练，加入是注意文件名格式.
from numpy import *
import operator
from os import listdir
import datetime
#1.得到最邻近K中的识别值
def getvalue(inX, dataSet, labels, k):
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

#2.将二值图片转换为1*1024的行向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
#进行手写数字的识别
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')          
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))#2034*1024的矩阵用于存放特征向量
    for i in range(m):
        fileNameStr = trainingFileList[i]                  
        fileStr = fileNameStr.split('.')[0]                
        classNumStr = int(fileStr.split('_')[0])          
        hwLabels.append(classNumStr)#得到所有图片的实际值
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
     
    testFileList = listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):#循环进行比较,得出准确率
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = getvalue(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    return errorCount/float(mTest)

     
print("\n演示使用KNN临近算法对手写数字的分类识别:\n")
starttime = datetime.datetime.now()
errors=handwritingClassTest()
endtime = datetime.datetime.now()
accuracyRate=(1-errors)*100
print("=====================KNN临近算法识别手写数字结果如下======================================================="+"="*20)
print("识别手写数字的准确率为: %f%%"%accuracyRate) 
usertime=(endtime - starttime).seconds
print("共用时:%d秒"%usertime)
print("数据来源:来Benjamin Wang的网络爬虫小I.")
print("======================================================================================================="+"="*20)
