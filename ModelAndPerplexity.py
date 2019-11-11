# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:14:36 2019

@author: Aydin
"""

import re
import math


def getSentences(file_path):
    with open(file_path, "r",encoding="UTF-8") as myfile:
        return [re.split(r"\s", i.strip('\n')) for i in myfile]
    #splitting lines (sentences for our case) with respect to spaces (regularExp \s),
    #thus getting seperate words. We also need to omit the newline expression since it has nothing to do with our language model 


 
#we can have different types of language models, start from unigram model

class Unigram:
    def __init__(self, lines):
        self.howOftenDictionary=dict()
        self.numOfWords=0
        self.numOfUniqueWords=0;
        for singleSentence in lines:
            for wd in singleSentence:
                self.howOftenDictionary[wd]=self.howOftenDictionary.get(wd,0) + 1
                if wd!='' and wd!='<s>' and wd!='</s>':
                    self.numOfWords +=1;
        self.numOfUniqueWords=len(self.howOftenDictionary)-2 #we do not count <s> and </s> as meaningful words
    def probOfUni(self,word):
        #Laplace Smoothing 
        unknown=0
        if self.howOftenDictionary.get(word,0)==0:
            unknown=1
            
        return float(self.howOftenDictionary.get(word,0)+1)/ float(self.numOfWords+self.numOfUniqueWords+unknown)
class Bigram(Unigram):
    def __init__(self,lines):
        Unigram.__init__(self,lines) #in order to compute the number of unique unigrams
        self.uniqueBigrams=set() #we use sets to prevent duplication of elements
        self.biFrequency=dict()
        
        for sent in lines:
            prev=None
            for w in sent:
                if prev !=None:
                    self.biFrequency[(prev,w)]=self.biFrequency.get((prev,w),0) + 1
                    self.uniqueBigrams.add((prev,w))
                prev=w
           
    def probOfBi(self,prevword,word):
        #Laplace Smoothing
        unknown=0
        if self.howOftenDictionary.get(word,0)==0:
            unknown=1
        return float(self.biFrequency.get((prevword,word),0)+1) / float(self.howOftenDictionary.get(prevword,0)+len(self.howOftenDictionary))

def perplexity_uni(trained,test):
    numberOfWords=0
    multiple=1
    for sent in test:
        for word in sent:
            if word !='<s>' and word !='</s>':
                numberOfWords+=1
                multiple*=trained.probOfUni(word)
                
    return math.pow(1/multiple,1/numberOfWords)

def perplexity_bi(trained, test):
    numberOfTokens=0
    multiple=1
    for sent in test:
        prev=None
        for word in sent:
            if word !='<s>':
                numberOfTokens+=1
            if prev!=None:
                multiple*=trained.probOfBi(prev,word)
            prev=word
    return math.pow(1/multiple,1/numberOfTokens)

        
    
if __name__ == '__main__':
    trainingModel=getSentences('./Train.txt')
    
    index=0
    flag=1
    
    for sent in trainingModel:
        
        for wd in sent:
            if(wd==''):
                flag=1
        if flag==1:
            trainingModel.pop(index)
        index+=1
        flag=0                
    testModel=getSentences('./Test.txt')
    testModel2=getSentences('./Test2.txt')
    testModel.pop(0)
    testModel[0].pop()
    testModel2.pop(0)

    
    
    
    dataToTrain = Bigram(trainingModel)

    print("perplexity of 'hə' in unigram model: ", perplexity_uni(dataToTrain, testModel))
    print("perplexity of 'hə' in bigram model:: ", perplexity_bi(dataToTrain, testModel))
    print("perplexity of 'deyin' in unigram model: ", perplexity_uni(dataToTrain, testModel2))
    print("perplexity of 'deyin' in bigram model: ", perplexity_bi(dataToTrain, testModel2))
