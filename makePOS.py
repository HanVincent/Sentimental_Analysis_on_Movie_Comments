import nltk
import json
from collections import defaultdict   

"""
pos_dict = defaultdict(list)
with open('pos_dict.json', 'r') as fp:
    pos_dict = json.load(fp)

print("Reading training data...")
data = list(open("train.tsv", "r"))[1:]
print("done!")

for index, d in enumerate(data):
    col = d.split('\t')
    data[index] = col[2].lower()
    
for eachphrase in data:
    eachphrase = nltk.word_tokenize(eachphrase)
    pos_phrase = nltk.pos_tag(eachphrase)
    for word in pos_phrase:
        if word[1] not in pos_dict[word[0]]:
            pos_dict[word[0]].append(word[1])

#################################################

print("Reading test data...")
test_data = list(open("test.tsv", "r"))[1:]
print("done!")

for index, d in enumerate(test_data):
    col = d.split('\t')
    test_data[index] = col[2].lower()

for eachphrase in test_data:
    eachphrase = nltk.word_tokenize(eachphrase)
    pos_phrase = nltk.pos_tag(eachphrase)
    for word in pos_phrase:
        if word[1] not in pos_dict[word[0]]:
            pos_dict[word[0]].append(word[1])

#################################################
            
with open("pos_dict.json", "w") as fp:
    json.dump(pos_dict, fp)
"""