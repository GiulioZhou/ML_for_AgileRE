import string, csv, numpy as np
from textLearning import tfidfColumn, tfidfAll
from sklearn.decomposition import PCA

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
	global dataset
	with open('prova.csv', 'rU') as csvfile:
		dataset = list(csv.reader(csvfile, skipinitialspace=True, delimiter =";"))
	# return getNumpyArray()

def applyPCA(number, feature_train, feature_test):
	pca = PCA(n_components=1, whiten=True).fit(feature_train)
	return pca.transform(feature_train), pca.transform(feature_test)


def processData(testBeginIndex, target):
	loadDataset()
	formatted_data = getNumpyArray()
	label_train = formatted_data[:testBeginIndex, np.newaxis, target]
	label_test = formatted_data[testBeginIndex:, np.newaxis, target]
	feature_train, feature_test = tfidfAll(dataset, testBeginIndex, label_train)

	for i in range (target+1,9):
		feature_train = np.concatenate((feature_train,formatted_data[:testBeginIndex,np.newaxis, i]), axis=1)
		feature_test = np.concatenate((feature_test,formatted_data[testBeginIndex:,np.newaxis, i]), axis=1)

	feature_train, feature_test = applyPCA(1,feature_train,feature_test)

	return feature_train, feature_test, label_train, label_test
