import numpy
import urllib
import scipy
import nltk
import random
import collections
from nltk.stem.porter import *
from nltk.corpus import stopwords

### function for calculate the most common sentiment

def getFrequent(value):#value:[1,4,2,2,5,3,2]
    dict_count = collections.defaultdict(int)#[1,2,3,4,5] 
    for v in value:
        dict_count[v] += 1
    common = max(dict_count, key = dict_count.get)#未排除相等數量, key=value.get return key
    return common
        
print("Reading data...")
data = list(open("../train.tsv", "r"))ㄋ
print("done!")

###record each word and its sentiment

dict_statistics = collections.defaultdict(list)

###grab each word's sentiment, stemming and lowercase

#stemmer = PorterStemmer()
for each in data[1:]:
    splitLine = each.split('\t')
    phrase = splitLine[2].split()
    #phrase = splitLine[2].lower().split()
    #phrase = [word for word in phrase if word not in stopwords.words('english')]
    for each_token in phrase:
        #stemmed_word = stemmer.stem(each_token)
        dict_statistics[each_token].append(splitLine[3].strip())

print(len(dict_statistics))
###real dictionary for sentiment
dict_token = collections.defaultdict(int)

###use function to calcu number and establish the real dictionary.

for key, value in dict_statistics.items():
    common = getFrequent(value)
    dict_token[key] = common
    
###write down the dictionary
    
f = open("dict.tsv", 'w')
for key, value in dict_token.items():
    f.write(key+","+value+"\n")
f.close()

###get the dictionary for each word's sentiment
