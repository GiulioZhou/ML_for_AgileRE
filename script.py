#!/usr/bin/python

import numpy as np
from sklearn import linear_model, svm, neighbors, tree

from dataExtractor import processData

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


#Clean away the 10% of points that have the largest residual errors
def outlierCleaner(predictions, features, targets):
	cleaned_data = []

	for count, elem in np.ndenumerate(predictions):
		error = predictions[count] - targets[count]
		cleaned_data.append((tuple(features[count[0]].tolist()), targets[count], error))

	cleaned_data.sort(key=lambda tup: tup[2])

	count = int(len(cleaned_data)*0.9)
	cleaned_data = cleaned_data[:count]
	return cleaned_data

#Display the chart (Note that you can't display more than 1 feature, or it is just me that I am not able to :D)
def display(feature_train, label_train, feature_test, label_test, pred):
	plt.clf()
	plt.scatter(feature_train, label_train, color="b", label="train data")
	plt.scatter(feature_test, label_test, color="r", label="test data")
	plt.plot(feature_test, pred, color="black", linewidth = 3)
	plt.legend(loc=2)
	plt.xlabel(ELABORATION)
	plt.ylabel(LOC)
	plt.show()

#Ordinary Least Squares
def testLinerRegression(trainx, trainy):
	regr = linear_model.LinearRegression()
	regr.fit(trainx, trainy)
	return regr

#Support Vector Machines
def testSVMRegression(trainx, trainy, k, c):
	regr = svm.SVR(kernel=k, C=c)
	regr.fit(trainx, trainy)
	return regr

#Nearest Neighbors regression
def testNeighborsRegression(trainx, trainy, n_neighbors, weights, ls):
	regr = neighbors.KNeighborsRegressor(n_neighbors, weights = weights, leaf_size = ls)
	regr.fit(trainx, trainy)
	return regr

#Naive Bayes regression
def testBayesRegression(trainx, trainy):
	from sklearn.naive_bayes import GaussianNB
	regr = GaussianNB()
	regr.fit(trainx, trainy)
	return regr

#Decision Trees regression
def testDTreeRegression(trainx, trainy, msl):
	regr = tree.DecisionTreeRegressor(min_samples_leaf=msl)
	regr.fit(trainx, trainy)
	return regr

if __name__ == "__main__":

	feature_train, feature_test, label_train, label_test = processData(100,LOC)

	# data = loadDataset()
	#
	# label_train = data[:100, np.newaxis, LOC]
	# label_test = data[100:, np.newaxis, LOC]
	# feature_train = data[:100, np.newaxis, EFFORT]
	# feature_test = data[100:, np.newaxis, EFFORT]

	#In case you want to use more features you can use the whole datase, removing the columns you don't need
	# data = np.delete(data,LOC,1) #delete the target column
	# data = np.delete(data,EXP_OUTPUT,1)
	# data = np.delete(data,DEF_DONE,1)
	# data = np.delete(data,ELABORATION,1)
	# data = np.delete(data,BUS_VALUE,1)
	# data = np.delete(data,USER_STORY,1)

	#feature_train = data[:100]
	#feature_test = data[100:]

	#reg = testLinerRegression(feature_train, label_train)
	#reg = testSVMRegression(feature_train, label_train, "linear", 300) #linear kernel with C= 300 makes a score of 0.71 (without the lengths)
	reg = testNeighborsRegression(feature_train, label_train, 25, "uniform", 15) # 0.7340
	#reg = testBayesRegression(feature_train, label_train)
	#reg = testDTreeRegression(feature_train, label_train, 3)

	pred = reg.predict(feature_test)

	#Removing some outliers has improved the score from -0.3596 to -0.1199
	#However, it seems that removing some data when considering all the features makes the score worse: 0.5354 to 0.0457
	cleaned_data = []#outlierCleaner(pred, feature_train, label_test)


	#Refit if the data has been cleaned
	if len(cleaned_data) > 0:
		feature_train, label_train, errors = zip(*cleaned_data)
		feature_train = np.array(feature_train)
		label_train = np.array(label_train)
		reg.fit(feature_train, label_train)
		pred = reg.predict(feature_test)


	#Using all the informations available to determine the LOC the score is 0.3828
	#While using only the user_story elaboration lenght the score is -0.3596 (which is quite bad)
	#Removing the lenght of business value lenght, and definition of done lenght the score is 0.5354
	print reg.score(feature_test, label_test)

	#display(feature_train, label_train, feature_test, label_test, pred)
