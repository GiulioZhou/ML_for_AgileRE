#!/usr/bin/python

import numpy as np
import xlsxwriter, itertools

from regression import testRegression
from dataExtractor import processData

from math import fabs
import matplotlib.pyplot as plt

def all_feature():
	features = []
	for i in range (1,7):
		features.append(feature_option[i](i))
	return features

def retElem(elem):
	return elem

def ask_text(elem):
	print "Select text fields. 0 for all"
	printText()
	selected = ask()
	if 0 in selected:
		selected = range(1,6)
	return (elem,selected)

def ask_numeric(elem):
	print "Select numerical fields. 0 for all"
	printNumerical()
	selected = ask()
	if 0 in selected:
		selected = range(1,7)
	return (elem,selected)


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


feature_option = {
	0: all_feature,
	1: ask_text,
	2: ask_numeric,
	3: ask_numeric,
	4: retElem,
	5: retElem,
	6: retElem,
}

def ask():
	ask = raw_input()
	return map(int, ask.split())

def retrieveFeatures():
	TRAINING_NUMBER = input("Choose the training set number (<=200)")
	if TRAINING_NUMBER <0 or TRAINING_NUMBER>200:
		return
	print "Choose a target"
	printNumerical()
	target = input()+4

	print "Select one or more features (separated by space), 0 for all the features"
	printFeatures()

	feature_string = ask()
	if 0 in feature_string:
		features = feature_option[0]()
	else:
		features = []
		for i in feature_string:
			features.append(feature_option[int(i)](int(i)))
	return features, TRAINING_NUMBER, target

def run():

	features, TRAINING_NUMBER, target = retrieveFeatures()
	feature_train, feature_test, label_train, label_test = processData(TRAINING_NUMBER,target, features)

	# print "Choose algorithm"
	# algorithm = input()
	reg = testRegression(feature_train, label_train.ravel(), 2)
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
		worksheet.write(i+1,2,fabs(prediction-effective))
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


def printNumerical():
	print "1-LOC"
	print "2-New Classes"
	print "3-Changed Classes"
	print "4-Effort"
	print "5-N. Tests"
	print "6-Entropy"
def printText():
	print "1-User Story"
	print "2-Business Value"
	print "3-Elaboration"
	print "4-Definition of done"
	print "5-Expected output"
def printFeatures():
	print "1-Length"
	print "2-Numerical values"
	print "3-Numerical history"
	print "4-Services"
	print "5-Word frequency"
	print "6-Semantic info"

def convertFeatures(f_tuple):
	features = []
	features.extend(((1,[]),(2,[]),(3,[])))
	for elem in f_tuple:
		if elem < 6:
			features[0][1].append(elem)
		if elem >5 and elem<12:
			features[1][1].append(elem-5)
		if elem >11 and elem<18:
			features[2][1].append(elem-11)
		if elem >17:
			features.append(elem-14)
	return features

#Compute all the combinations of features
#It's tooking too long to find all the scores (1048576 combinations)
def automaticTests():
	TRAINING_NUMBER = 50
	target = 5 #LOC
	#Save all the prediction in a xlsx file
	workbook = xlsxwriter.Workbook('./partial_results/%s'%target +'_%s'%TRAINING_NUMBER+'.xlsx')
	worksheet = workbook.add_worksheet()
	line = 1
	worksheet.write(0,0,"Features")
	worksheet.write(0,1,"Score")
	for L in range(0, len(max_features)+1):
	    for subset in itertools.combinations(max_features, L):
			features = convertFeatures(subset)
			print features
			feature_train, feature_test, label_train, label_test = processData(TRAINING_NUMBER,target, features)
			if feature_train.size != 0:
				reg = testRegression(feature_train, label_train.ravel(), 2)
				pred = reg.predict(feature_test)
				score = reg.score(feature_test, label_test)
				print score
				worksheet.write(line,0,str(features))
				worksheet.write(line,1,score)
				line += 1
	workbook.close()

def autoFold():
	#features = [(1,[1,2,3,4,5]),(2,[1,2,3,4,5,6]),(3,[1,2,3,4,5,6]),4,5,6]
	features = [(2,[2,3]),5]
	for target in range(10,11):
		TRAINING_NUMBER = 200
		#Save all the prediction in a xlsx file
		workbook = xlsxwriter.Workbook('./partial_results/%s'%target +'.xlsx')
		worksheet = workbook.add_worksheet()
		j=0
		print target
		while TRAINING_NUMBER>40:
			TRAINING_NUMBER -= 25
			feature_train, feature_test, label_train, label_test = processData(TRAINING_NUMBER,target, features)

			reg = testRegression(feature_train, label_train.ravel(), 2)
			pred = reg.predict(feature_test)
			score = reg.score(feature_test, label_test)
			print score
			worksheet.write(0,j*4+0,"Pred")
			worksheet.write(0,j*4+1,"Effective")
			worksheet.write(0,j*4+2,"Error")
			worksheet.write(0,j*4+3,"T_Number")
			worksheet.write(1,j*4+3,TRAINING_NUMBER)
			worksheet.write(4,j*4+3,"Score")
			worksheet.write(5,j*4+3,score)
			err_sum=0
			for i in range(200-TRAINING_NUMBER):
				effective=label_test[i]
				prediction=int(reg.predict(feature_test[i].reshape(1,-1))[0])
				worksheet.write(i+1,j*4+0,prediction)
				worksheet.write(i+1,j*4+1,effective)
				err = fabs(prediction-effective)
				worksheet.write(i+1,j*4+2,err)
				err_sum += err

			worksheet.write(2,j*4+3,"Average error")
			worksheet.write(3,j*4+3,int(err_sum/(200-TRAINING_NUMBER)))
			j+=1
		workbook.close()
		return

if __name__ == "__main__":
	autoFold()
	#run()
