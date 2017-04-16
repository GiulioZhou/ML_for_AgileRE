import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

# Returns the text with all the stem of the words in it
def textStemmer(text):
	#Remove punctuation
	text_string = text.translate(string.maketrans("", ""), string.punctuation)
	words = text_string.split()
	stemmer = SnowballStemmer("english")
	stemmed = ""
	for word in words:
		if is_ascii(word): #Some words have awkward characters in it and I don't know why so I just ignore them
			stemmed = stemmed + stemmer.stem(word) + " "
	return stemmed

#Extract the stems of the words in a column
def tfidfColumn(dataset, testBeginIndex, index, labels_train):
    word_data = []
    for i, elem in enumerate(dataset):
        words = textStemmer(dataset[i][index])
        word_data.append(words)

    #Vectorise the words
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(word_data[:testBeginIndex]).toarray()
    features_test_transformed  = vectorizer.transform(word_data[testBeginIndex:]).toarray()

    #Feature selection
    # selector = SelectPercentile(f_classif, percentile=1)
    # selector.fit(features_train_transformed, labels_train)
    # features_train_transformed = selector.transform(features_train_transformed)
    # features_test_transformed  = selector.transform(features_test_transformed)

    return features_train_transformed.toarray(), features_test_transformed.toarray()
