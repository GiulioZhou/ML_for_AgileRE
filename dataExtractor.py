import ast, string, csv, numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from textLearning import tfidfColumn, tfidfAll, getSemanticVector

def concatenate(first, second):
	return np.concatenate((first, second), axis=1)

def loadDataset():
	global dataset
	with open('dataset.csv', 'rU') as csvfile:
		dataset = list(csv.reader(csvfile, skipinitialspace=True, delimiter =";"))

#Principal Components Analysis
def applyPCA(number, feature_train, feature_test):
	pca = PCA(n_components=1, whiten=True).fit(feature_train)
	return pca.transform(feature_train), pca.transform(feature_test)

#Reduce the dimension of the array by selecting a percentile of the total features
def doFeatureSelection (feature_train, feature_test, labels_train):
	selector = SelectPercentile(f_classif, percentile=1)
	selector.fit(feature_train, labels_train)
	return selector.transform(feature_train), selector.transform(feature_test)

#Count the words in a field
def charCounter(row, column):
	return len((str(dataset[row][column])).split())

def getNumeric(row, column):
	return int(dataset[row][column+4])

#Sum of the previous values
def totalHistory(row,column):
	sum=0
	for user_story in dataset[:row]:
		sum += int(user_story[column+4])
	return sum

def servicesToList(row):
	list_services = ast.literal_eval(dataset[row][11]) #string to list
	services = [0]*(15*15)
	for i, elem in enumerate(list_services):
		services[i*15:i*15+len(list_services[i])] = list_services[i]
	return services

#Calculate the interesting data for a user story
def getData(row, features_list, target):
	result = []
	for elem in features_list:
		if isinstance(elem, tuple):
			for column in elem[1]:
				if not (elem[0] != 1 and column == (target-4)):
					result.append(getColumn[elem[0]](row, column))
	if 4 in features_list:
		result += servicesToList(row)
	return result

getColumn = {
	1: charCounter,
	2: getNumeric,
	3: totalHistory,
	4: servicesToList,
}

def column(matrix, i):
    return [int(row[i]) for row in matrix]

def processData(testBeginIndex, target, features_list):
	#Load data from CVS
	loadDataset()
	data = []
	for count, user_story in enumerate(dataset):
		data.append(getData(count, features_list, target))
	formatted_data = np.asarray(data)

	if 5 in features_list:
		#Retrieve statistical information
		feature_train, feature_test = tfidfAll(dataset, testBeginIndex)
		feature_train = concatenate(formatted_data[:testBeginIndex], feature_train)
		feature_test = concatenate(formatted_data[testBeginIndex:],feature_test)

	if 6 in features_list:
		#Retrieve semantical information
		x,y = getSemanticVector(dataset, testBeginIndex)
		feature_train = concatenate(feature_train,x)
		feature_test = concatenate(feature_test,y)

	feature_train, feature_test = applyPCA(1,feature_train,feature_test)

 	label_train = np.asarray(column(dataset[:testBeginIndex], target))
	label_test = np.asarray(column(dataset[testBeginIndex:], target))

	return feature_train, feature_test, label_train, label_test
