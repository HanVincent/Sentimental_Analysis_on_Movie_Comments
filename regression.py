# -*- coding: utf-8 -*-
"""
Try to use Regression to classify the sentimental analysis.
y = x * theta
"""
import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import string
from nltk.stem.porter import *
from sklearn import linear_model

### Read the training data

print("Reading data...")
data = list(open("train.tsv", "r"))[0:5000]
print("done")

### Process the data and the data structure

p_data = defaultdict(int)
for d in data[1:]:
    col = d.split('\t')
    p_data[col[2]] = float(col[3].strip())

### How many unique words are there?

wordCount = defaultdict(int)
for k, v in p_data.items():#get each instance
  for w in k.split():#using space to split each word
    wordCount[w] += 1#count the number of words

print("lenth of original wordCount:", len(wordCount))#how many different words

### Ignore capitalization and remove punctuation

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for k, v in p_data.items():
  r = ''.join([c for c in k.lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

print("length of punctuation: ", len(wordCount))

### With stemming

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stemmer = PorterStemmer()
for k, v in p_data.items():
  r = ''.join([c for c in k.lower() if not c in punctuation])
  for w in r.split():
    w = stemmer.stem(w)
    wordCount[w] += 1
print("length of stem: ", len(wordCount))

### Just take the most popular words...先拿前一千名

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for k, v in p_data.items():
  r = ''.join([c for c in k.lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]#just get the words to list

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))#會依照字母出現頻率做index, a:0, the:1...
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum.lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(k) for k, v in p_data.items()]#每個instance都會有自己的向量
y = [v for k, v in p_data.items()]#y = answer(rank)

#No regularization
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
#predictions = numpy.round(clf.predict(X))

### Read test data

print("Reading data...")
test_data = list(open("test.tsv", "r"))[1:]
print("done")

test_X = [feature(d.split('\t')[2]) for d in test_data]

predictions = numpy.round(clf.predict(test_X))
print(predictions)

fw = open("predict.csv", "w")
fw.write("PhraseId,Sentiment\n")
for index in range(len(test_data)):
    fw.write(test_data[index].split('\t')[0])
    fw.write(",")
    
    if predictions[index] < 0:
        fw.write("0")
    elif predictions[index] > 4:
        fw.write("4")
    else:
        fw.write(str(int(predictions[index]))) 
        
    fw.write("\n")
fw.close()