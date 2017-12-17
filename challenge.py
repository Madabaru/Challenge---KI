#!/usr/bin/env python3

"""

@author: adminmac

"""
# Import dependecies 
import re
import nltk
import numpy as np

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
train_neg = np.array(open('train-neg.txt', 'r').read().split('\n'))
train_pos = np.array(open('train-pos.txt', 'r').read().split('\n'))
test = np.array(open('test.txt', 'r').read().split('\n'))

# Preprocessing training data
train_neg = np.stack((train_neg, np.zeros((1500), dtype = np.int8)), axis = 1)
train_pos = np.stack((train_pos, np.ones((1500), dtype = np.int8)), axis = 1)
test = np.stack((test, np.ones((1000), dtype = np.int8)), axis = 1)
data = np.vstack((train_neg, train_pos, test))

corpus = []
ps = PorterStemmer()
for i in range(0, 4000):
    review = ''.join(map(str, data[[i], 0])).split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Converting words to features
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y_train = data[:3000, 1]
X_train = X[:3000, :]
X_test = X[-1000:, :]

# Classifying
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)




# if __name__ == '__main__':
 