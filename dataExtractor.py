import ast, string, csv, numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_regression, RFE, chi2
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

#select the feature with higher score
def fs_percentile (feature_train, feature_test, labels_train):
	selector = SelectPercentile(f_regression, percentile=10)
	selector.fit(feature_train, labels_train)
	return selector.transform(feature_train), selector.transform(feature_test)

#Count the words in a field
def charCounter(row, column):
	return len((str(dataset[row][column-1])).split())

def getNumeric(row, column):
	return int(dataset[row][column+4])

#Sum of the previous values
def totalHistory(row,column):
	#sum= int(dataset[row][column+4])
	sum = 0
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
	try:
		dataset
	except:
		loadDataset()
	data = []

	if features_list == [(1,[1,2,3,4,5]),(2,[1,2,3,4,5,6]),(3,[1,2,3,4,5,6]),4,5,6]:
		try:
			formatted_data = np.load("./npy/all_feature%s"%target+".npy")
		except:
			for count, user_story in enumerate(dataset):
				data.append(getData(count, features_list, target))
			formatted_data = np.asarray(data)
			np.save("./npy/all_feature%s"%target,formatted_data)
	else:
		for count, user_story in enumerate(dataset):
			data.append(getData(count, features_list, target))
		formatted_data = np.asarray(data)

	feature_train = formatted_data[:testBeginIndex]
	feature_test = formatted_data[testBeginIndex:]

	if 5 in features_list:
		#Retrieve statistical information
		try:
			x=np.load("./npy/tfidf%s"%testBeginIndex+"x.npy")
			y=np.load("./npy/tfidf%s"%testBeginIndex+"y.npy")
		except:
			x,y = tfidfAll(dataset, testBeginIndex)
			np.save("./npy/tfidf%s"%testBeginIndex+"x",x)
			np.save("./npy/tfidf%s"%testBeginIndex+"y",y)

		feature_train = concatenate(feature_train,x)
		feature_test = concatenate(feature_test,y)

	if 6 in features_list:
		#Retrieve semantical information
		x,y = getSemanticVector(dataset, testBeginIndex)
		feature_train = concatenate(feature_train,x)
		feature_test = concatenate(feature_test,y)

	label_train = np.asarray(column(dataset[:testBeginIndex], target))
	label_test = np.asarray(column(dataset[testBeginIndex:], target))



	#print feature_train.size
	#feature_train, feature_test = fs_percentile (feature_train, feature_test, label_train)

	# if feature_train != []:
	# 	feature_train, feature_test = applyPCA(1,feature_train,feature_test)
	return feature_train, feature_test, label_train, label_test
