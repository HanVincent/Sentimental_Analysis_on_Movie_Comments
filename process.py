from collections import defaultdict
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

def processData(dataset, select):
  temp_data = []
  ### How many unique words are there?
  if select == "1":
    for datum in dataset:
      temp_data.append(datum)
    print("Do nothing.")

  ### Ignore capitalization and remove punctuation
  elif select == "2":
    punctuation = set(string.punctuation)
    for datum in dataset:
      line = ''.join([c for c in datum.lower() if not c in punctuation])
      temp_data.append(line)
    print("Ignore capitalization and remove punctuation")

  ### With stemming
  elif select == "3":
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    for datum in dataset:
      line = ''.join([c for c in datum.lower() if not c in punctuation])  
      line = ' '.join([stemmer.stem(w) for w in line.split()])
      temp_data.append(line)
    print("With stemming")

  ### Remove Stopwords
  elif select == "4":
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    for datum in dataset:
      line = ''.join([c for c in datum.lower() if not c in punctuation]) 
      r_words = [word for word in line.split() if word not in stopwords.words('english')]
      line = ' '.join([stemmer.stem(w) for w in r_words])
      temp_data.append(line)    
    print("Remove stopwords")
  
  ### Just lowercase and remove stopwords
  elif select == "5":
    punctuation = set(string.punctuation)
    for datum in dataset:
      lower = ''.join([c for c in datum.lower() if not c in punctuation])
      line = ' '.join([word for word in lower.split() if word not in stopwords.words('english')])      
      temp_data.append(line)
    print("Lowercase and Remove punct. and stopwords.")
    
  return temp_data

###########################################################################
  
def countWord(dataset, select):
  wordCount = defaultdict(int)
  
  ### Unigram
  if select == "1":
    for datum in dataset:
      for w in datum.split():
        wordCount[w] += 1
    print("Unigram: ", len(wordCount))

  ### Bigram
  elif select == "2":
    for datum in dataset:
      words = datum.split()
      for index in range(len(words)-1):
        w = ' '.join([words[index], words[index+1]])
        wordCount[w] += 1
    print("Bigram: ", len(wordCount))

  ### Trigram
  elif select == "3":
    for datum in dataset:
      words = datum.split()
      for index in range(len(words)-2):
        w = ' '.join([words[index], words[index+1], words[index+2]])
        wordCount[w] += 1
    print("Trigram: ", len(wordCount))

  ### Unigram + Bigram
  elif select == "4":
    for datum in dataset:
      words = datum.split()
      for index in range(len(words)-1):
        w = ' '.join([words[index], words[index+1]])
        wordCount[w] += 1 #bigram
        wordCount[words[index]] #unigram
      if len(words) > 0:
        wordCount[words[len(words)-1]] += 1  #unigram - last word
    print("Unigram + Bigram: ", len(wordCount))

  ### Unigram + Bigram + Trigram    
  elif select == "5":
    for datum in dataset:
      words = datum.split()
      for index in range(len(words)-2):
        w = ' '.join([words[index], words[index+1], words[index+2]])
        wordCount[w] += 1 #trigram
        w = ' '.join([words[index], words[index+1]])
        wordCount[w] += 1 #bigram
        wordCount[words[index]] += 1 #unigram
        
      if len(words) > 1: #bigram - last pair
        w = ' '.join([words[len(words)-2], words[len(words)-1]]) 
        wordCount[w] += 1 
        
      if len(words) > 0: #unigram - last two words
        wordCount[words[len(words)-2]] += 1  
        wordCount[words[len(words)-1]] += 1
      
  return wordCount

###########################################################################
  

"""
if __name__ == '__main__':
    # test1.py executed as script
    # do something
    processData()
    """