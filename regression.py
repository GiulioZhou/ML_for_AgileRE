from sklearn import linear_model, svm, neighbors, tree

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

regression = {
    1: testLinerRegression,
    2: testSVMRegression,
    3: testNeighborsRegression,
    4: testBayesRegression,
    5: testDTreeRegression,
}

def testRegression(feature_train, label_train, algorithm,):
    #TODO generalise the parameters
    return regression[algorithm](feature_train,label_train, "linear", 10)
    #reg = testLinerRegression(feature_train, label_train)
    #reg = testSVMRegression(feature_train, label_train.ravel(), "linear", 10)
    #reg = testNeighborsRegression(feature_train, label_train, 25, "uniform", 15)
    #reg = testBayesRegression(feature_train, label_train)
    #reg = testDTreeRegression(feature_train, label_train, 3)
