import ast, string, csv, numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from textLearning import tfidfColumn, tfidfAll, getSemanticVector


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

#convert the services matrix into a numpy array
def servicesToArray(row):
	list_services = ast.literal_eval(dataset[row][11]) #string to list
	services= np.zeros(15*15, dtype=np.int)
	for i, elem in enumerate(list_services):
		service = np.asarray(elem)
		services[i*15:i*15+service.shape[0]] = service
	return services

def servicesToList(row):
	list_services = ast.literal_eval(dataset[row][11]) #string to list
	services = [0]*(15*15)
	for i, elem in enumerate(list_services):
		services[i*15:i*15+len(list_services[i])] = list_services[i]
	return services

#Calculate the interesting data for a user story
def getData(row):
	result = []
	for i in range(0,5):
		result.append(charCounter(row, i))
	for i in range(5,11):
		#result.append(totalHistory(row, i))
		result.append(int(dataset[row][i]))
	result += servicesToList(row)
	return tuple(result)

#Provide the formatted data of the dataset
def getNumpyArray():
	result = []
	for count, user_story in enumerate(dataset):
		result.append(getData(count))
	return np.asarray(result)

def loadDataset():
	global dataset
	with open('dataset.csv', 'rU') as csvfile:
		dataset = list(csv.reader(csvfile, skipinitialspace=True, delimiter =";"))
	# return getNumpyArray()

def applyPCA(number, feature_train, feature_test):
	pca = PCA(n_components=1, whiten=True).fit(feature_train)
	return pca.transform(feature_train), pca.transform(feature_test)

def doFeatureSelection (feature_train, feature_test, labels_train):
	#Feature selection -> reduce the dimension of the array. However, since we don't have
	#a big dataset, I believe that it is not really necessary
	selector = SelectPercentile(f_classif, percentile=1)
	selector.fit(feature_train, labels_train)
	return selector.transform(feature_train), selector.transform(feature_test)


def processData(testBeginIndex, target):
	#Load data from CVS
	loadDataset()
	#Retrieve the numerical data
	formatted_data = getNumpyArray()
 	label_train = formatted_data[:testBeginIndex, np.newaxis, target]
	label_test = formatted_data[testBeginIndex:, np.newaxis, target]

	#Retrieve statistical information
	feature_train, feature_test = tfidfAll(dataset, testBeginIndex)

	for i in range (5,11+(15*15)):
		if i != target:
			feature_train = np.concatenate((feature_train,formatted_data[:testBeginIndex,np.newaxis, i]), axis=1)
			feature_test = np.concatenate((feature_test,formatted_data[testBeginIndex:,np.newaxis, i]), axis=1)

	#Retrieve semantical information
	x,y = getSemanticVector(dataset, testBeginIndex)
	feature_train = np.concatenate((feature_train,x), axis=1)
	feature_test = np.concatenate((feature_test,y), axis=1)

	feature_train, feature_test = applyPCA(1,feature_train,feature_test)

	return feature_train, feature_test, label_train, label_test
