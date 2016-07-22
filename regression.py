# -*- coding: utf-8 -*-
"""
Try to use Regression to classify the sentimental analysis.
y = x * theta
"""
import numpy
from process import *
from collections import defaultdict
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import linear_model

### Read the training data
print("Reading data...")
data = list(open("train.tsv", "r"))[1:]
print("done")

### Process the data structure
p_data = []
p_data_v = []
for d in data:
    col = d.split('\t')
    p_data.append(col[2])
    p_data_v.append(float(col[3].strip()))
    
### Process
print("1.Original\n2.Lowercase and remove punct.\n3.plus stemming\n4.remove stopwords")
select = input("Select: ")
p_data = processData(p_data, select)

### Gram
print("1.Unigram\n2.Bigram\n3.Trigram\n4.Unigram + Bigram\n5.Unigram + Bigram + Trigram\n")
gram_select = input("Select n-gram: ")
wordCount = countWord(p_data, gram_select)

### Sort in order
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

#======================================================================#
### Pick more effective words
words = [x[1] for x in counts[:1000]]
print(words)
#======================================================================#

### Sentiment analysis
wordId = dict(zip(words, range(len(words))))# a:0, the:1...

def feature(datum, gram_select):
  feat = [0] * len(words)

  ### Unigram
  if gram_select == "1":
    for w in datum.split():
      if w in words:
        feat[wordId[w]] += 1
        
  ### Bigram
  if gram_select == "2":
    w_split = datum.split()    
    for index in range(len(w_split)-1):
      w = ' '.join([w_split[index], w_split[index+1]])
      if w in words:
        feat[wordId[w]] += 1
        
  ### Trigram
  if gram_select == "3":
    w_split = datum.split()
    for index in range(len(w_split)-2):
      w = ' '.join([w_split[index], w_split[index+1], w_split[index+2]])
      if w in words:
        feat[wordId[w]] += 1
      
  ### Unigram + Bigram
  if gram_select == "4":
    w_split = datum.split()
    for index in range(len(w_split)-1):
      w = ' '.join([w_split[index], w_split[index+1]])
      if w in words:
        feat[wordId[w]] += 1
      if w_split[index] in words:
        feat[wordId[w_split[index]]] += 1
    if len(w_split) > 0 and w_split[len(w_split)-1] in words:
      feat[wordId[w_split[len(w_split)-1]]] += 1
      
  ### Unigram + Bigram + Trigram
  if gram_select == "5":
    w_split = datum.split()
    for index in range(len(w_split)-2):
      w = ' '.join([w_split[index], w_split[index+1], w_split[index+2]]) #Trigram
      if w in words: 
        feat[wordId[w]] += 1
      w = ' '.join([w_split[index], w_split[index+1]]) #Bigram
      if w in words: 
        feat[wordId[w]] += 1
      w = w_split[index] #Unigram
      if w in words:
        feat[wordId[w]] += 1
    
    if len(w_split) > 1: #Last bigram term
      w = ' '.join([w_split[len(w_split)-2], w_split[len(w_split)-1]])
      if w in words:
          feat[wordId[w]] += 1

    if len(w_split) > 0: #Last unigram term
      if w_split[len(w_split)-2] in words:
        feat[wordId[w_split[len(w_split)-2]]] += 1 
      if w_split[len(w_split)-1] in words:
        feat[wordId[w_split[len(w_split)-1]]] += 1
 
  feat.append(1)     
  return feat
  
### Setting y = X * theta
X = [feature(datum, gram_select) for datum in p_data]
y = [v for v in p_data_v]

### Training
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_

### Read test data
print("Reading test data...")
test_data = list(open("test.tsv", "r"))[1:]
print("done")

p_test_data = []
for d in test_data:
    col = d.split('\t')
    p_test_data.append(col[2])

### Process  
p_test_data = processData(p_test_data, select)
  
### Comparing string
test_X = [feature(datum, gram_select) for datum in p_test_data]

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
