# -*- coding: utf-8 -*-
# @Time    : 2/26/2018 9:31 PM
# @FileName: temp.py
# Info: unittest
import KNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
np.savetxt("allData.txt", datingDataMat)
np.savetxt("label.txt", datingLabels)
plt.show()

