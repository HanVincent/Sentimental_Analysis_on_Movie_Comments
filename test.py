import numpy
import urllib
import scipy
import random
import collections
from nltk.stem.porter import *

print("Reading data...")
data = list(open("test.tsv"))
print("done!")

#read the dictionary
dictionary = collections.defaultdict(int)
f = open("dict.tsv", 'r')

for eachline in f:
    a = eachline.split(',')
    dictionary[a[0]] = a[1]

f.close()
    
#start to calcu
stemmer = PorterStemmer()
fw = open("predict.csv", 'w')
fw.write("PhraseId,Sentiment\n")
for each in data[1:]:
    split = each.split('\t')
    phrase = split[2]
    totalScore = 2
    for each_token in phrase.lower().split():
        stem_token = stemmer.stem(each_token)
        if stem_token in dictionary:      
            totalScore += int(dictionary[stem_token])
        else:
            totalScore += 2
    
    totalScore = 2 if len(phrase.split()) == 0 else totalScore / len(phrase.split())
        
    fw.write(split[0] + "," + str(round(totalScore)) + "\n")
    
fw.close()
    