import numpy
import urllib
import scipy
import random
import collections

#
def getFrequent(value):#value:[1,4,2,2,5,3,2]
    dict_count = collections.defaultdict(int)    
    for v in value:
        dict_count[v] += 1
    maximum = max(dict_count, key = dict_count.get)#未排除相等數量
    return maximum
        

print("Reading data...")
data = list(open("train.tsv", "r"))
print("done!")

dict_statistics = collections.defaultdict(list)#record each word and its sentiment

for each in data[1:]:
    phrase = each.split('\t')[2]
    for each_token in phrase.split():
        dict_statistics[each_token].append(each.split('\t')[3].strip())

dict_token = collections.defaultdict(int)
for key, value in dict_statistics.items():
    maximum = getFrequent(value)
    dict_token[key] = maximum#取得最後dictionary.
    
f = open("dict.tsv", 'w')
for key, value in dict_token.items():
    f.write(key+","+value+"\n")
f.close()
#get the dictionary for each word's sentiment
