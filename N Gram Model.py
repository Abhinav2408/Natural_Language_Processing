import math
import random
import re
import string
import time
from sklearn.linear_model import LinearRegression
import numpy as np

sentences = []
vocab = {}

tokenizedSentences = []
unktokenizedSentences = []

for i in range(6):

    textfile = open('Book' + str(i+1)+ '.txt','r',encoding='utf-8')
    data = textfile.read()
    textfile.close()
    data = data.lower()
    data = re.sub('\n', '', data)
    data = re.sub(r'[^\w\s.,]', '', data)
    data = re.sub('j.k. rowling','.jk rowling.', data)
    data = re.sub('mr.','mr', data)
    data = re.sub('mrs.','mrs', data)
    sentences.extend(data.split('.'))


def splitter(sentence: str) -> list[str]:
    for p in string.punctuation:
        sentence = sentence.replace(p, ' ' + p + ' ')
    return sentence.split()

for i in range(len(sentences)):

    if len(sentences[i])>0:
        sentences[i] += '.'
    
    x = splitter(sentences[i])
    for token in x:
        if token in vocab:
            vocab[token] +=1
        else:
            vocab[token] =1
    tokenizedSentences.append(x)



class Nmodel(object):

    def __init__(self,number):
        self.n = number
        self.gramfreq = {}
        self.context = {}
        self.countofGramfreq = {}

    def ngrammaker(self,tokens: list) -> list:
        #add n-1 starting tags

        tokens = (self.n-1)*['<tag>'] + tokens
        thislist = []
        for i in range(self.n - 1, len(tokens)):
            prevwords = [tokens[i-self.n+1+j] for j in range(self.n-1)]
            nthword = tokens[i]
            thislist.append((tuple(prevwords),nthword))

        return thislist

    def frequency(self, sentence: list) -> None:
        allngrams = self.ngrammaker(sentence)
        for gram in allngrams:
            if gram in self.gramfreq:
                self.gramfreq[gram] += 1
            else:
                self.gramfreq[gram] = 1    
            
            if gram[0] in self.context:
                self.context[gram[0]].append(gram[1])
            else:
                self.context[gram[0]] = [gram[1]]

        for gram in allngrams:
            if self.gramfreq[gram] in self.countofGramfreq:
                self.countofGramfreq[self.gramfreq[gram]] +=1
            else:
                self.countofGramfreq[self.gramfreq[gram]] =1

    def setTuring(self) -> None:
        x = np.array(list(self.countofGramfreq.keys()))
        y = list(self.countofGramfreq.values())

        linreg = LinearRegression().fit(np.log(x.reshape(-1,1)),np.log(y))
        a = linreg.intercept_
        b = linreg.coef_.item()

        for i in range(1,max(x)+1):
            if i not in self.countofGramfreq:
                self.countofGramfreq[i] = math.exp(a + b*(math.log(i)))

        self.countofGramfreq[0] = sum(self.gramfreq.values())



    def harrypotter(self,tokenlen : int):
        prevwords = (self.n-1)*['<tag>']
        text = []

        while (tokenlen>0) and ((len(text)==0) or (text[len(text)-1]!='.')):
            
            usedword = ''
            try:
                possiblewords = self.context[tuple(prevwords)]
            except KeyError:
                break

            tokenToProb = {}
            for i in range(len(sorted(possiblewords))):
                word = possiblewords[i]
                try:
                    wordcnt = self.gramfreq[(tuple(prevwords),word)]
                    prevwordscnt = float(len(self.context[tuple(prevwords)]))
                    prob = (wordcnt)/(prevwordscnt)
                    i += wordcnt-1
                except KeyError:
                    prob = 0.0
                tokenToProb[word] = prob

            random.seed(time.time())
            r = random.random()
            val = 0

            for word in tokenToProb:
                val += tokenToProb[word]
                if val>=r:
                    usedword = word
                    break
            text.append(usedword)
            if self.n>1:
                prevwords.pop(0)
                if usedword == '.':
                    prevwords = (self.n-1)*['<tag>']
                else:
                    prevwords.append(usedword)
            tokenlen -=1
        return ' '.join(text)




ngramModels = []

for i in range(1,9):

    model = Nmodel(i)
    for sentence in tokenizedSentences:
        model.frequency(sentence)
    model.setTuring()        
    print("n = " + str(i))
    print(model.harrypotter(30))
    ngramModels.append(model)




    ###############     Testing     ##############



testfile = open('Book' + str(7)+ '.txt','r',encoding='utf-8')
testdata = testfile.read()
testfile.close()
testdata = testdata.lower()
testdata = re.sub('\n', '', testdata)
testdata = re.sub(r'[^\w\s.,]', '', testdata)
testdata = re.sub('j.k. rowling','.jk rowling.', testdata)

testsentences = testdata.split('.')
tokenizedtestSentences = []


for i in range(len(testsentences)):

    if len(testsentences[i])>0:
        testsentences[i] += '.'
    
    x = splitter(testsentences[i])
    tokenizedtestSentences.append(x)




##############   Kneser Function   ###############



def knesersolver(n : int,idx : int,prevwords: list, word: string, discount : int, storedprobs : dict):

    if (tuple(prevwords[idx:]),word) in storedprobs:
        return storedprobs[(tuple(prevwords[idx:]),word)]

    model = ngramModels[n-1]
    if n==1:
        try:
            return (float(vocab[word])/(sum(vocab.values())))
        except KeyError:
            return discount*(1.0/len(vocab))

    else:
        try:
            wordcnt = model.gramfreq[(tuple(prevwords[idx:]),word)]
            prevwordscnt = float(len(model.context[tuple(prevwords[idx:])]))
            term = (max(wordcnt - discount,0.0))/prevwordscnt
        except KeyError:
            term = 0

        try:
            prevwordscnt = float(len(model.context[tuple(prevwords[idx:])]))
            unique = len(np.unique(np.array(model.context[tuple(prevwords[idx:])])))
            lamda = (discount*unique)/prevwordscnt
        except KeyError:
            return discount*(1.0/len(vocab))

        term += lamda*knesersolver(n-1,idx+1,prevwords,word,discount,storedprobs)
        storedprobs[(tuple(prevwords[idx:]),word)] = term
        return term




##############   Backoff Function   ###############



def backoffsolver(n : int,idx : int,prevwords: list, word: string, storedprobs : dict):

    if (tuple(prevwords[idx:]),word) in storedprobs:
        return storedprobs[(tuple(prevwords[idx:]),word)]

    model = ngramModels[n-1]
    if n==1:
        try:
            return math.log(vocab[word]/(sum(vocab.values())))
        except KeyError:
            return -math.log(len(vocab))

    else:
        try:
            wordcnt = model.gramfreq[(tuple(prevwords[idx:]),word)]
            prevwordscnt = float(len(model.context[tuple(prevwords[idx:])]))
            prob = math.log((wordcnt)/(prevwordscnt))
        except KeyError:
            prob = math.log(0.4) + backoffsolver(n-1,idx + 1,prevwords,word,storedprobs)

        storedprobs[(tuple(prevwords[idx:]),word)] = prob
        return prob




##############   Linear Interpolation Function   ###############



def interpolationsolver(n : int,idx : int,prevwords: list, word: string, storedprobs: dict):

    if (tuple(prevwords[idx:]),word) in storedprobs:
        return storedprobs[(tuple(prevwords[idx:]),word)]

    model = ngramModels[n-1]
    if n==1:
        try:
            return (float(vocab[word])/(sum(vocab.values())))
        except KeyError:
            return (1.0/len(vocab))

    else:
        try:
            wordcnt = model.gramfreq[(tuple(prevwords[idx:]),word)]
            prevwordscnt = float(len(model.context[tuple(prevwords[idx:])]))
            prob = float((wordcnt)/(prevwordscnt))
        except KeyError:
            prob = 0.0

        prob += (n-1)*interpolationsolver(n-1,idx+1,prevwords,word,storedprobs)
        prob = prob/n
        storedprobs[(tuple(prevwords[idx:]),word)] = prob
        return prob




# Using Dynamic Programming to speed up Interpolation, Backoff and Kneser ney


interpolationProbabs = {}
kneserProbabs = {}
backoffProbabs = {}




################   Probability of a sentence  ################



def logprobab(n : int,sentence : list, smoothing : string):

    tokens = (n-1)*['<tag>'] + sentence
    model = ngramModels[n-1]

    if smoothing == 'add one':
        if n==1:
            logprob = 0.0
            for word in sentence:
                try:
                    logprob += math.log(vocab[word]/(sum(vocab.values())))
                except KeyError:
                    logprob -= math.log(len(vocab))
            return logprob

        else:
            logprob = 0.0
            for i in range(n-1, len(tokens)):
                prevwords = tokens[i-n+1:i]
                word = tokens[i]
                try:
                    wordcnt = model.gramfreq[(tuple(prevwords),word)]
                    prevwordscnt = float(len(model.context[tuple(prevwords)]))
                    prob = (1 + wordcnt)/(prevwordscnt + len(vocab))
                except KeyError:
                    try:
                        prevwordscnt = float(len(model.context[tuple(prevwords)]))
                        prob = (1)/(prevwordscnt + len(vocab))
                    except KeyError:
                        prob = 1.0/len(vocab)

                logprob += math.log(prob)

            return logprob


    elif smoothing=='good turing':
        logprob = 0.0
        for i in range(n-1,len(tokens)):
            prevwords = tokens[i-n+1:i]
            word = tokens[i]
            try :
                c = (model.gramfreq[(tuple(prevwords),word)])
            except KeyError:
                c = 0
            cstar = float(c)
            if c<max(model.countofGramfreq.keys()):
                cstar = (float(c+1)*model.countofGramfreq[c+1])/model.countofGramfreq[c]

            try : 
                divisor = math.log(len(model.context[tuple(prevwords)]))
            except KeyError:
                divisor = math.log(model.countofGramfreq[0]) + math.log(model.countofGramfreq[c])
            logprob += math.log(cstar) - divisor

        return logprob


    elif smoothing == 'kneser ney':

        discount = 0.75

        if n==1:
            logprob = 0.0
            for word in sentence:
                try:
                    logprob += math.log(vocab[word]/(sum(vocab.values())))
                except KeyError:
                    logprob -= math.log(len(vocab))
            return logprob

        else:
            logprob = 0.0
            for i in range(n-1, len(tokens)):
                prevwords = tokens[i-n+1:i]
                word = tokens[i]
                logprob += math.log(knesersolver(n,0,prevwords,word,discount,kneserProbabs))     
            return logprob




    elif smoothing == 'backoff':
        if n==1:
            logprob = 0.0
            for word in sentence:
                try:
                    logprob += math.log(vocab[word]/(sum(vocab.values())))
                except KeyError:
                    logprob -= math.log(len(vocab))
            return logprob

        else:
            logprob = 0.0
            for i in range(n-1, len(tokens)):
                prevwords = tokens[i-n+1:i]
                word = tokens[i]
                logprob += (backoffsolver(n,0,prevwords,word,backoffProbabs))     
            return logprob


    elif smoothing == 'interpolation':

        #taking all lambdas as 1/n

        if n==1:
            logprob = 0.0
            for word in sentence:
                try:
                    logprob += math.log(vocab[word]/(sum(vocab.values())))
                except KeyError:
                    logprob -= math.log(len(vocab))
            return logprob

        else:
            logprob = 0.0
            for i in range(n-1, len(tokens)):
                prevwords = tokens[i-n+1:i]
                word = tokens[i]
                logprob += math.log(interpolationsolver(n,0,prevwords,word,interpolationProbabs))     
            return logprob


############       Perplexity of sentences       ############


def perplexity(n : int,sentences : list, smoothing : string):
    m = 0
    logprob = 0.0
    for sentence in sentences:
        m+= len(sentence)
        logprob -= logprobab(n,sentence,smoothing)
    m += (n-1)*(len(sentences))
    logprob = logprob/m
    return math.exp(logprob)


################         Main Testing          #################


for smoothing in ['add one', 'good turing', 'backoff' , 'interpolation', 'kneser ney']:
    for model in ngramModels:
        print("For n =",model.n, "and smoothing:",smoothing)
        print(perplexity(model.n,tokenizedtestSentences,smoothing))