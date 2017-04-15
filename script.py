import numpy as np

from dataExtractor import loadDataset

import matplotlib.pyplot as plt

USER_STORY = 0
BUS_VALUE = 1
ELABORATION = 2
DEF_DONE = 3
EXP_OUTPUT = 4
LOC = 5
N_CLASS = 6
C_CLASS = 7
EFFORT = 8
TEST_REV = 9
ENTROPY = 10
SERVICES = 11


def testLinerRegression(trainx, trainy):
	from sklearn import linear_model

	regr = linear_model.LinearRegression()
	regr.fit(trainx, trainy)
	
	return regr


if __name__ == "__main__":
	data = loadDataset()
	
	feature_train = data[:100, np.newaxis, ELABORATION]
	feature_test = data[100:, np.newaxis, ELABORATION]
	label_train = data[:100, np.newaxis, LOC]
	label_test = data[100:, np.newaxis, LOC]
	
	reg = testLinerRegression(feature_train, label_train)
	pred = reg.predict(feature_test)
	print pred
	
	plt.clf()
	plt.scatter(feature_train, label_train, color="b", label="train data")
	plt.scatter(feature_test, label_test, color="r", label="test data")
	plt.plot(feature_test, pred, color="black")
	plt.legend(loc=2)
	plt.xlabel(ELABORATION)
	plt.ylabel(LOC)
	plt.show()
	













