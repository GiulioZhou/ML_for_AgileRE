#!/usr/bin/python

import numpy as np
import xlsxwriter, itertools
from math import fabs

from regression import testRegression
from dataExtractor import processData

#brief description of the features
#1 -> words counter for 1-user story, 2-business value, 3 etc....
#2 -> numerical value for 1-LOC, 2-New calsses, 3 etc...
#3 -> sum of the previous numerical value for 1-LOC etc...
#4 -> services
#5 -> tfidf
#6 -> semantic -> doc2vec
def run():
	#all the features
	#features = [(1,[1,2,3,4,5]),(2,[1,2,3,4,5,6]),(3,[1,2,3,4,5,6]),4,5,6]
	features = [(1,[1,2,3,4,5]),5,6]

	for target in range(5,11):
		TRAINING_NUMBER = 200
		#Save all the prediction in a xlsx file
		workbook = xlsxwriter.Workbook('./partial_results/%s'%target +'.xlsx')
		worksheet = workbook.add_worksheet()
		j=0
		print target
		while TRAINING_NUMBER>40:
			TRAINING_NUMBER -= 25
			feature_train, feature_test, label_train, label_test = processData(TRAINING_NUMBER,target, features)

			reg = testRegression(feature_train, label_train.ravel(), 2) #2 for svm
			pred = reg.predict(feature_test)
			score = reg.score(feature_test, label_test)
			print score

			#Write on the xlsx file
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

if __name__ == "__main__":
	run()
