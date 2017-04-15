import csv, numpy as np

#Count the words in a field
def charCounter(row, column):
	return len((str(dataset[row][column])).split())

#Sum of the previous values
def totalHistory(row,column):
	if column < 5 or column > 10:
			return -1
	sum=0
	for user_story in dataset[:row]:
		sum += int(user_story[column])
	return sum

#Calculate the interesting data for a user story
def getData(row):
	result = []
	for i in range(0,5):
		result.append(charCounter(row, i))
	for i in range(5,9):
#		result.append(totalHistory(row, i))
		result.append(int(dataset[row][i]))
	return tuple(result)

#Provide the formatted data of the dataset
def getNumpyArray():
	result = []
	for count, user_story in enumerate(dataset):
		result.append(getData(count))
	return np.asarray(result)

def loadDataset():
	with open('prova.csv', 'rU') as csvfile:
		global dataset
		dataset = list(csv.reader(csvfile, skipinitialspace=True, delimiter =";"))
		return getNumpyArray()









