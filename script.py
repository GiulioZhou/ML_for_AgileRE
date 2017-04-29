#!/usr/bin/python

import numpy as np
import xlsxwriter
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
	regr = svm.SVR(kernel=k, C=c,gamma='auto')
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

def run():
	TRAINING_NUMBER = input("Choose the training set number (<=200)")
	if TRAINING_NUMBER <0 or TRAINING_NUMBER>200:
		return
	print "Choose a target"
	print "1-LOC"
	print "2-New Classes"
	print "3-Changed Classes"
	print "4-Effort"
	print "5-N. Tests"
	print "6-Entropy"
	target = input()+4

	feature_train, feature_test, label_train, label_test = processData(TRAINING_NUMBER,target)

	#reg = testLinerRegression(feature_train, label_train)
	reg = testSVMRegression(feature_train, label_train.ravel(), "linear", 10)
	#reg = testNeighborsRegression(feature_train, label_train, 25, "uniform", 15)
	#reg = testBayesRegression(feature_train, label_train)
	#reg = testDTreeRegression(feature_train, label_train, 3)
	pred = reg.predict(feature_test)

	#Not so useful so far
	cleaned_data = []#outlierCleaner(pred, feature_train, label_test)

	#Refit if the data has been cleaned
	if len(cleaned_data) > 0:
		feature_train, label_train, errors = zip(*cleaned_data)
		feature_train = np.array(feature_train)
		label_train = np.array(label_train)
		reg.fit(feature_train, label_train)
		pred = reg.predict(feature_test)

	#Score -> (1- u/v) where u is the regression sum of squares, and v is the residual sum of squares
	# u= sum((test-pred)^2) v= sum((test-testMean)^2)
	print "R^2 score (1.0 is the best score)"
	score = reg.score(feature_test, label_test)
	print score

	#Save all the prediction in a xlsx file
	workbook = xlsxwriter.Workbook('./partial_results/%s'%target +'_%s'%TRAINING_NUMBER+'.xlsx')
	worksheet = workbook.add_worksheet()
	worksheet.write(0,0,"Pred")
	worksheet.write(0,1,"Effective")
	worksheet.write(0,2,"Error")
	worksheet.write(0,5,"Score")
	worksheet.write(1,5,score)

	for i in range(200-TRAINING_NUMBER):
		effective=label_test[i]
		prediction=int(reg.predict(feature_test[i].reshape(1,-1))[0])
		worksheet.write(i+1,0,prediction)
		worksheet.write(i+1,1,effective)
		worksheet.write(i+1,2,prediction-effective)
	workbook.close()

	print "Predictions saved!"

	#Provide the prediction for a specific user story
	# while True:
	# 	number = input("Choose a user story 50-199, -1 to exit\n")
	# 	if number == -1 or number <TRAINING_NUMBER or number >199:
	# 		break
	# 	else:
	# 		print int(reg.predict(feature_test[number-TRAINING_NUMBER])[0].reshape(-1,1))

	#display(feature_train, label_train, feature_test, label_test, pred)

if __name__ == "__main__":
	while True:
		run()
