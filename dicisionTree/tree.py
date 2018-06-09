# -*- coding: utf-8 -*-
# @Time    : 4/24/2018 7:06 PM
# @FileName: tree.py
# Info: chapter 3. decision tree

from math import log
import operator
from dicisionTree import treePlotter


# calculate dataSet entropy
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # calculate each symbol's probability
        prob = float(labelCounts[key]) / numEntries
        # calculate  the entropy
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# column1: can survive without coming to surface
# column2: has flippers
# column3: Fish?
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    return dataSet, label


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFectVec = featVec[:axis]
            reducedFectVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFectVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if infoGain > bestInfoGain:  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# 多数表决函数
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# ataSet = [[1, 1, 'yes'],
#            [1, 1, 'yes'],
#            [1, 0, 'no'],
#            [0, 1, 'no'],
#            [0, 1, 'no']]
# label = ['no surfacing', 'flippers']
# 先找出bestFeat，是column1, 从上面的dataSet可以看到，如果column1是0，则一定是‘no’, 如果是1， 则需要进一步判断column2
# create出来的tree是以dict的形式存储的: {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def createTree(dataSet, labels):
    # create tree的时候是需要column 3里的结果的
    classList = [example[-1] for example in dataSet]
    # 第一个中止条件是所有类的标签相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # stop splitting when there are no more features in dataSet
    # column1 column2的特征值都被删除掉了，只剩下column3 -> 多数决策
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        # call splitDataSet,根据bestFeat将dataSet分成两个数据集，分别createTree
        # -> 如果所有类的标签相同，即可认为根据该bestFeat可以将该子dataSet区分出来
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 一个问题是程序无法确定特征在inputTree的位置
# 第一个是生成的决策树，参数二用于确定特征在数据集中的位置，参数三是待决策的输入数据
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    print(classify(myTree, labels, [1, 1]))
