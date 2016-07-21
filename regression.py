# -*- coding: utf-8 -*-
"""
Try to use Regression to classify the sentimental analysis.
y = x * theta
"""
import numpy
from collections import defaultdict
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import linear_model

### Read the training data

print("Reading data...")
data = list(open("train.tsv", "r"))
print("done")

### Process the data and the data structure

p_data = defaultdict(int)
for d in data[1:]:
    col = d.split('\t')
    p_data[col[2]] = float(col[3].strip())

print("1.Original")
print("2.Lowercase and remove punct.")
print("3.plus stemming")
print("4.remove stopwords")
select = input("Select: ")

wordCount = defaultdict(int)

### How many unique words are there?
if select == "1":
  for k, v in p_data.items():
    for w in k.split():
      wordCount[w] += 1
  print("lenth of original wordCount:", len(wordCount))

### Ignore capitalization and remove punctuation
elif select == "2":
  punctuation = set(string.punctuation)
  for k, v in p_data.items():
    r = ''.join([c for c in k.lower() if not c in punctuation])
    for w in r.split():
      wordCount[w] += 1
  print("length of punctuation: ", len(wordCount))

### With stemming
elif select == "3":
  punctuation = set(string.punctuation)
  stemmer = PorterStemmer()
  for k, v in p_data.items():
    r = ''.join([c for c in k.lower() if not c in punctuation])  
    for w in r.split():
      w = stemmer.stem(w)
      wordCount[w] += 1
  print("length of stem: ", len(wordCount))

### Remove Stopwords
elif select == "4":
  punctuation = set(string.punctuation)
  stemmer = PorterStemmer()
  for k, v in p_data.items():
    r = ''.join([c for c in k.lower() if not c in punctuation]) 
    r_words = [word for word in r.split() if word not in stopwords.words('english')]
    for w in r_words:
      w = stemmer.stem(w)
      wordCount[w] += 1
  print("length of removing stopwords: ", len(wordCount))


### Just take 1000 words to train
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

#======================================================================#
### Pick more effective words
words = [x[1] for x in counts[:1000]]
print(words)
#======================================================================#

### Sentiment analysis
wordId = dict(zip(words, range(len(words))))#會依照字母出現頻率做index, a:0, the:1...
wordSet = set(words)

def feature(datum, select):
  feat = [0] * len(words)
    
  ### Original  
  if select == "1":
    for w in datum.split():
      if w in words:
        feat[wordId[w]] += 1
  
  ### Lowercase and remove punctuation  
  elif select == "2":
    r = ''.join([c for c in datum.lower() if not c in punctuation])
    for w in r.split():
      if w in words:
        feat[wordId[w]] += 1
    
  ### Stemming
  elif select == "3":  
    r = ''.join([c for c in datum.lower() if not c in punctuation])
    for w in r.split():
      w = stemmer.stem(w)
      if w in words:
        feat[wordId[w]] += 1
    
  ### Remove stopwords
  elif select == "4":  
    r = ''.join([c for c in datum.lower() if not c in punctuation])
    r_words = [word for word in r.split() if word not in stopwords.words('english')]
    for w in r_words:
      w = stemmer.stem(w)
      if w in words:
        feat[wordId[w]] += 1
 
  feat.append(1)     
  return feat
    
X = [feature(k, select) for k, v in p_data.items()]#每個instance都會有自己的向量
y = [v for k, v in p_data.items()]#y = answer(rank)

### Training
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_

### Read test data
print("Reading test data...")
test_data = list(open("test.tsv", "r"))[1:]
print("done")

test_X = [feature(d.split('\t')[2], select) for d in test_data]

### Predict
predictions = numpy.round(clf.predict(test_X))

### Write file
fw = open("predict.csv", "w")
fw.write("PhraseId,Sentiment\n")
for index in range(len(test_data)):
    fw.write(test_data[index].split('\t')[0] + ",")
    
    if predictions[index] < 0:
        fw.write("0")
    elif predictions[index] > 4:
        fw.write("4")
    else:
        fw.write(str(int(predictions[index]))) 

    fw.write("\n")
fw.close()