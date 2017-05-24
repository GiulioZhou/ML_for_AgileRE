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
