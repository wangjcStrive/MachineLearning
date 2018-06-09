# -*- coding: utf-8 -*-
# @Time    : 5/22/2018 2:14 PM
# @FileName: bayes.py
# Info:

from numpy import *
from numpy.ma import log


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 筛选掉dataSet里的重复word
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# inputSet的元素如果在vocabList里，则在vocabList的位置上置为1
# 将一组单词转化为一组数字
# 如果inputSet里的word再vocabList里存在，则在存在的位置上置1
# @vacabList: 不重复的所有的word
# @inputSet: 待转化成数值型的输入
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 基于贝叶斯准则，计算文档词条属于各个类的概率，然后比较概率大小，得出分类结果
# p(c|w) = p(w|c)p(c)/p(w)
#   p(c0|w): 在word w出现的情况下，是c0的概率
#   p(w|c0): 已知是c0(非abuse)的情况下出现该word的概率
# 训练算法，从词向量计算概率p(w0|ci)...及p(ci)
# @trainMatrix：由每篇文档的词条向量组成的文档矩阵。输入时6句话（document），是在所有vocabulary里该word是否出现
#   trainMatrx = [1, 0, 1, 0,...1]
#                [0, 0, 1, 0,...0]
#                [1, 1, 0, 1,...1]
#                [0, 0, 1, 0,...0]
#                [0, 0, 1, 1,...0]
#                [0, 0, 1, 0,...1]
#
#   先根据trainCategory[0,1,0,1,0,1]分成2组
#   把以上matrix各行相加得到P1Num:[2, 1, 5, 2,...3],然后除以所有word的数量{比如p1Denom=30}可以得到:
#   p1Vect = [2/30, 1/30, 5/30, ...], 即函数的返回值
# @trainCategory:每篇文档的类标签组成的向量
#   这里要考虑到条件概率。以下函数计算的是条件概率，即，在条件trainCategory下某个word出现的概率
#
# 返回值：p1Vect：在已知为abuse的条件下，每个单词出现的概率
#   p1Vect = [P(word1|C1), P(word2|C1), P(word3|C1).....]
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = [P(word1|C1), P(word2|C1), P(word3|C1).....]
    p1Vect = log(p1Num/p1Denom)  # change to log()
    p0Vect = log(p0Num/p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# 判断一句话是否是abuse的。而不是判断一个word是否为abuse
# 在已知某个单词的条件下，该单词是abuse的。即求p(w|c1)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p(w|c1)*p(c1)
    # sum(vec2Classify*p1Vec)是这句话的期望?
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)  # log(a*b) = log(a) + log(b)
    # p(w1, w2, w3....|c0)*p(c0)其中每个word是相互独立的 => p(w1|c0) * p(w2|c0) * p(w3|c0) *...
    # 再取对数
    p0 = sum(vec2Classify*p0Vec) + log(1 - pClass1)
    print("p0:%d, p1:%d" % (p0, p1))
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = list(range(50))
    testSet = []  # create test set
    # delete 10 elements
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 输入的是,trainMat:
    # [0, 2, 1, 0, 0, 4....]
    # [5, 0, 0, 0, 1, 3....]
    # trainClasses:
    # [1, 0, 1, 0, 0, 1....]
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    # return vocabList,fullText


if __name__ == '__main__':
    spamTest()
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print(p0V)
    print(p1V)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('%s classified as: %d' %
          (testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('%s classified as: %d' %
          (testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
