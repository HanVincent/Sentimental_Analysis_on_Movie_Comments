import numpy
import urllib
import scipy
import random
import collections
from nltk.stem.porter import *
from nltk.corpus import stopwords

print("Reading data...")
data = list(open("test.tsv"))
print("done!")

###read the dictionary and store in dict

dictionary = collections.defaultdict(int)
fr = open("dict.tsv", 'r')
for eachline in fr:
    a = eachline.split(',')
    dictionary[a[0]] = a[1]
fr.close()
    
###start to calculate each phrase
    
stemmer = PorterStemmer()
fw = open("predict.csv", 'w')
fw.write("PhraseId,Sentiment\n")
for each in data[1:]:
    totalScore = 0
    splitLine = each.split('\t')
    #phrase = splitLine[2].split()
    phrase = splitLine[2].lower().split()
    phrase = [word for word in phrase if word not in stopwords.words('english')]
    for each_token in phrase:
        #stem_token = each_token        
        stem_token = stemmer.stem(each_token)
        if stem_token in dictionary:      
            totalScore += int(dictionary[stem_token])
        else:
            totalScore += 2
            
    totalScore = 2 if len(phrase) == 0 else totalScore / len(phrase)
        
    fw.write(splitLine[0] + "," + str(round(totalScore)) + "\n")
    
fw.close()
    