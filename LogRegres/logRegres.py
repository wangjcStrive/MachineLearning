# -*- coding: utf-8 -*-
# @Time    : 6/6/2018 3:22 PM
# @FileName: logRegres.py
# Info: 

from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMatrix = []
    fr = open(r'testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMatrix.append(int(lineArr[2]))
    return dataMat, labelMatrix


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 前面已经求出了系数，在这里用
    # sigmoid的输入z = w0x0 + w1x1 + w2x2,其中0是分界线. 故设定0=w0x0 + w1x1 + w2x2,从而得到了X1，X2的关系式
    #y = arange(-weights[0]-weights[1]*x)/weights[2]
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# y = w1*x1 + w2*x2 + .. + wn*xn. 其中x是feature，而w是系数。
# 现在要求y的最大值，这取决于系数w，从而现在要确定w.
# 从而上述函数可看做变量为w的函数，进而根据梯度上升公式w = w + alpha * f'(w)
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)              # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        # 假设txt文件里是x,y的值，这里把x,y的值相加后代入到sigmoid函数里计算
        # 即，当回归系数都是1时候代入sigmoid计算

        # 梯度上升法每次更新回归系数都需要遍历整个数据集(dataMatrix)
        inputSigmoid = dataMatrix*weights
        h = sigmoid(inputSigmoid)
        error = (labelMat - h)
        # 梯度上升公式，看做是w的函数，对w求梯度
        # 每次更新回归系数时都需要遍历整个数据集
        weights = weights + alpha * dataMatrix.transpose() * error
    # 输入是100*3的矩阵，有3个feature,所以返回的是3*1的矩阵，作用于3个feature的系数
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    dataArr = array(dataMatrix)
    for i in range(m):
        # 这里与普通梯度上升法不同，普通的梯度上升这里是dataMatrix*weight
        h = sigmoid(sum(dataMatrix[i]*weights))
        # h = sigmoid(dataMatrix[i] + weights)
        error = classLabels[i] - h
        weights = weights + alpha*error*dataArr[i]
    return weights


# 我们期望算法能避免来回波动，从而收敛到某个值，并期望收敛加快
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    dataArr = array(dataMatrix)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha调整，缓解波动
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0


def colictest():
    frTrain = open("horseColicTraining.txt")
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeight = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCoutn = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        linArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeight)) != int(currLine[21]):
            errorCoutn += 1
    errorRate = (float(errorCoutn)/numTestVec)
    print('error rate is: %f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colictest()
    print('afeter %d interations the average error rate is: %f' % (numTests, errorSum/float(numTests)))



if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    weightsRandom = stocGradAscent0(dataArr, labelMat)
    weightsRandomAdvance = stocGradAscent1(dataArr, labelMat)
    # plotBestFit(weights.getA())
    # plotBestFit(weightsRandom)
    plotBestFit(weightsRandomAdvance)