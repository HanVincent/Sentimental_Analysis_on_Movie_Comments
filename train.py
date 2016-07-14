import numpy
import urllib
import scipy
import nltk
import random
import collections
from nltk.stem.porter import *

#
def getFrequent(value):#value:[1,4,2,2,5,3,2]
    dict_count = collections.defaultdict(int)    
    for v in value:
        dict_count[v] += 1
    common = max(dict_count, key = dict_count.get)#未排除相等數量
    return common
        

print("Reading data...")
data = list(open("train.tsv", "r"))
print("done!")

dict_statistics = collections.defaultdict(list)#record each word and its sentiment

#grab each word's sentiment and lowercase
stemmer = PorterStemmer()
for each in data[1:]:
    phrase = each.split('\t')[2].lower()
    for each_token in phrase.split():
        dict_statistics[stemmer.stem(each_token)].append(each.split('\t')[3].strip())

dict_token = collections.defaultdict(int)#real dictionary for sentiment

for key, value in dict_statistics.items():
    common = getFrequent(value)
    dict_token[key] = common#establish the real dictionary.
    
f = open("dict.tsv", 'w')
for key, value in dict_token.items():
    f.write(key+","+value+"\n")
f.close()
#get the dictionary for each word's sentiment
