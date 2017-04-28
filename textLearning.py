import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import numpy as np

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
        if is_ascii(word) and word not in stopwords.words('english'): #Some words have awkward characters in it and I don't know why so I just ignore them
            stemmed = stemmed + stemmer.stem(word) + " "
    return stemmed

def wordsToVector(word_data,testBeginIndex):
    #Vectorise the words
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(word_data[:testBeginIndex])
    features_test_transformed  = vectorizer.transform(word_data[testBeginIndex:])

    return features_train_transformed.toarray(), features_test_transformed.toarray()

#Extract the stems of the words in a column and vectorize the results
def tfidfColumn(dataset, testBeginIndex, index):
    word_data = []
    for i, elem in enumerate(dataset):
        words = textStemmer(dataset[i][index])
        word_data.append(words)

    return wordsToVector(word_data,testBeginIndex)

#Since getting the statistics of a single column did not give any interesting information,
#Let's try with all the (text) columns together
def tfidfAll(dataset, testBeginIndex):
    word_data = []
    words = ""
    for i, elem in enumerate(dataset):
        for j in range(5):
            words = words + textStemmer(dataset[i][j])
        word_data.append(words)
        words = ""
    return wordsToVector(word_data, testBeginIndex)

class TaggedLineSentence(object):
    def __init__(self, doc_list):
       self.doc_list = doc_list
    def __iter__(self):
        i = -1
        for idx, doc in enumerate(self.doc_list):
            i += 1
            if i == 5:
                i = 0
            #print '%s' %i +'_%s' % (idx/5)
            yield TaggedDocument(words=doc.split(),tags=['%s' %i +'_%s' % (idx/5)])


def getSemanticVector(dataset):
    #Load the vocabulary if it was previously created
    try:
        model = Doc2Vec.load("doc2vec.model")
    except:
        data = []
        for i, elem in enumerate(dataset):
            for j in range(5):
                data.append(textStemmer(dataset[i][j]))
        labeled = TaggedLineSentence(data)
        model = Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) #parametri a caso presi da un tutorial
        model.build_vocab(labeled)
        for epoch in range(50):
            model.train(labeled,total_examples=model.corpus_count,epochs=model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            model.train(labeled,total_examples=model.corpus_count,epochs=model.iter)

        model.save("doc2vec.model")

    feature_train=[]
    feature_test=[]
    for i in range(50):
        feature_train.append(model.docvecs['0_%s' %i])
        for j in range(1,5):
            feature_train[i] = np.concatenate((feature_train[i], model.docvecs['%s'%j+'_%s' %i]), axis=0)
    for i in range(150):
        feature_test.append(model.docvecs['0_%s' %i])
        for j in range(1,5):
            feature_test[i] = np.concatenate((feature_test[i], model.docvecs['%s'%j+'_%s' %(i+50)]), axis=0)

    # test = textStemmer("As a client, I want to play in the casino and have a tons of games displayed in the main UI")
    # new_vector = model.infer_vector(test)
    # print model.docvecs.most_similar([new_vector])

    return np.asarray(feature_train), np.asarray(feature_test)
