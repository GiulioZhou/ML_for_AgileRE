from sklearn import linear_model, svm, neighbors, tree

#Ordinary Least Squares
def testLinerRegression(trainx, trainy):
	regr = linear_model.LinearRegression()
	regr.fit(trainx, trainy)
	return regr

#Support Vector Machines
def testSVMRegression(trainx, trainy):
	regr = svm.SVR(kernel="linear", C=10 ,gamma='auto')
	regr.fit(trainx, trainy)
	return regr

def testSVMClassifier(trainx, trainy):
	clf = svm.SVC(C=1)
	clf.fit(trainx, trainy)
	return clf

#Nearest Neighbors regression
def testNeighborsRegression(trainx, trainy):
	regr = neighbors.KNeighborsRegressor( 25, weights = "uniform", leaf_size = 15)
	regr.fit(trainx, trainy)
	return regr

#Naive Bayes regression
def testBayesRegression(trainx, trainy):
	from sklearn.naive_bayes import GaussianNB
	regr = GaussianNB()
	regr.fit(trainx, trainy)
	return regr

#Decision Trees regression
def testDTreeRegression(trainx, trainy):
	regr = tree.DecisionTreeRegressor(min_samples_leaf=3)
	regr.fit(trainx, trainy)
	return regr

regression = {
    1: testLinerRegression,
    2: testSVMRegression,
    3: testNeighborsRegression,
    4: testBayesRegression,
    5: testDTreeRegression,
	6: testSVMClassifier
}

def testRegression(feature_train, label_train, algorithm):
	return regression[algorithm](feature_train,label_train)
