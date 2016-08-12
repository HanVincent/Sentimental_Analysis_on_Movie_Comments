# -*- coding: utf-8 -*-
from process import *
from grad import *

"""
Process & Gram
1.Original    2.Lowercase and remove punct.   3.plus stemming
4.remove stopwords   5.Lowercase and Remove stopwords

1.Unigram  2.Bigram  3.Trigram
4.Unigram + Bigram
5.Unigram + Bigram + Trigram
"""

newFeature = 3
selects = range(2, 3)
gram_selects = range(1, 2)
number = range(5000, 6000, 1000)
for select in selects:
    ################# TRAINING DATA ####################
    # Read the training data
    print("Reading training data...")
    data = list(open("train.tsv", "r"))[1:]
    print("done")
    
    # Transform the data structure
    p_data = []
    p_data_v = []
    for d in data:
        col = d.split('\t')
        p_data.append(col[2])
        p_data_v.append(float(col[3].strip()))
        
    # Process data and N-gram
    p_data = processData(p_data, select)
    
    ################# TEST DATA ####################
    # Read test data
    print("Reading test data...")
    test_data = list(open("test.tsv", "r"))[1:]
    print("done")
        
    # Transform the data structure
    p_test_data = []
    for d in test_data:
        col = d.split('\t')
        p_test_data.append(col[2])
        
    # Process data 
    p_test_data = processData(p_test_data, select)

    ################# COUNT N-GRAM ################# 
    for gram_select in gram_selects:
        wordCount = countWord(p_data, gram_select)
        counts = [(wordCount[w], w) for w in wordCount]
        counts.sort()
        counts.reverse()
            
        ################# MAKE DICTIONARY #################
        for n in number:
            """ 
            words = []
            for word in counts:
                eachWord = word[1].split()
                if len(eachWord) == 3:
                    if eachWord[0] in stopwords.words('english') and eachWord[1] in stopwords.words('english') and eachWord[2] in stopwords.words('english'):
                        continue
                    words.append(word)
                elif len(eachWord) == 2:
                    if eachWord[0] in stopwords.words('english') and eachWord[1] in stopwords.words('english'):   
                        continue
                    words.append(word)
                elif len(eachWord) == 1:
                    if eachWord[0] in stopwords.words('english'):
                        continue
                    words.append(word)
                
            words = [x[1] for x in words[:n]]
            """
            words = [x[1] for x in counts[:n]]
            """            
            words = []
            for word in counts:
                if word[1] in pos_dict:
                    if "JJ" in pos_dict[word[1]] or "NN" in pos_dict[word[1]]:
                        words.append(word[1])
                    
                if len(words) >= n:
                    break
            """
            ################# CREATE FEATURE #################
            wordId = dict(zip(words, range(len(words))))# a:0, the:1...
       
            X = [feature(datum, gram_select, wordId) for datum in p_data]
            y = [v for v in p_data_v]
            
            ################# TRAINING #################
            theta = [0] * (n + newFeature)
            theta = scipy.optimize.fmin_l_bfgs_b(f, theta, fprime, args = (X, y, 0))
            theta = theta[0]
            print(theta)
            
            ################# PREDICTING #################
            test_X = [feature(datum, gram_select, wordId) for datum in p_test_data]
            
            predictions = predict(test_X, theta)
            
            ################# WRITE FILE #################
            name = "Results/" + str(select) + "_"+ str(gram_select) +"_" + str(n) + ".csv"
            fw = open(name, "w")
            fw.write("PhraseId,Sentiment\n")
            for index in range(len(test_data)):
                fw.write(test_data[index].split('\t')[0] + ",")
                   
                if predictions[index] < 0:
                    fw.write("0")
                elif predictions[index] > 4:
                    fw.write("4")
                else:
                    fw.write(str(int(round(predictions[index]))))
                
                fw.write("\n")
            fw.close()
        
            # dictionary(model)
            dict_name = "dictionary/" + str(select) + "_"+ str(gram_select) +"_" + str(n) + ".csv"
            fw = open(dict_name, "w")
            for index in range(len(words)):
                fw.write(words[index])
                fw.write(",")
                fw.write(str(theta[index]))
                fw.write("\n")
            fw.close()
