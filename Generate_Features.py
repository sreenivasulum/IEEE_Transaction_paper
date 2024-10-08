'''
Created on 02-Feb-2017

@author: Koustav
'''

import sys
import re
import codecs
import string
import os
import numpy as np
from textblob import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gzip
from happyfuntokenizing import *
import numpy as np

lmtzr = WordNetLemmatizer()
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)

PRONOUN_PATH = '../Lexical_Resources/classifier_dictionary/english_pronoun.txt'
WHWORD_PATH = '../Lexical_Resources/classifier_dictionary/english_whwords.txt'
SLANG_PATH = '../Lexical_Resources/classifier_dictionary/english_swear.txt'
INTENSIFIER_PATH = '../Lexical_Resources/classifier_dictionary/english_intensifier.txt'
SUBJECTIVE_PATH = '../Lexical_Resources/classifier_dictionary/subjclueslen1-HLTEMNLP05.tff'
EVENT_PATH = '../Lexical_Resources/classifier_dictionary/english_nonsituational_phrase.txt'
MODAL_VERB_PATH = '../Lexical_Resources/classifier_dictionary/english_modal_verb.txt'
RELIGION_PATH = '../Lexical_Resources/classifier_dictionary/communal_race.txt'

TAGGER_PATH = '../Lexical_Resources/ark-tweet-nlp-0.3.2/'

PRONOUN = {}
WHWORD = {}
SLANG = {}
INTENSIFIER = {}
SUBJECTIVE = {}
EVENT = {}
MODAL = {}
RELIGION = {}

def READ_FILES():

	fp = open(PRONOUN_PATH,'r')
        for l in fp:
		PRONOUN[l.strip(' \t\n\r').lower()] = 1
        fp.close()
	
	fp = open(INTENSIFIER_PATH,'r')
        for l in fp:
		INTENSIFIER[l.strip(' \t\n\r').lower()] = 1
        fp.close()
	
	fp = open(WHWORD_PATH,'r')
        for l in fp:
		WHWORD[l.strip(' \t\n\r').lower()] = 1
        fp.close()
	
	fp = open(SLANG_PATH,'r')
        for l in fp:
		SLANG[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	fp = open(EVENT_PATH,'r')
        for l in fp:
		EVENT[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	fp = open(MODAL_VERB_PATH,'r')
        for l in fp:
		MODAL[l.strip(' \t\n\r').lower()] = 1
        fp.close()
	
	fp = open(RELIGION_PATH,'r')
        for l in fp:
		RELIGION[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	fp = open(SUBJECTIVE_PATH,'r')
        for l in fp:
                wl = l.split()
                x = wl[0].split('=')[1].strip(' \t\n\r')
                if x=='strongsubj':
                        y = wl[2].split('=')[1].strip(' \t\n\r')
                        SUBJECTIVE[y.lower()] = 1
	fp.close()

############################ This Functions are used #############################################

def emoticons(s):
        return len(re.findall(u'[\U0001f600-\U0001f60f\U0001f617-\U0001f61d\U0001f632\U0001f633\U0001f638-\U0001f63e\U0001f642\U0001f646-\U0001f64f\U0001f612\U0001f613\U0001f615\U0001f616\U0001f61e-\U0001f629\U0001f62c\U0001f62d\U0001f630\U0001f631\U0001f636\U0001f637\U0001f63c\U0001f63f-\U0001f641\U0001f64d]', s))

def smileys(s):
        return len(re.findall(r':\-\)|:[\)\]\}]|:[dDpP]|:3|:c\)|:>|=\]|8\)|=\)|:\^\)|:\-D|[xX8]\-?D|=\-?D|=\-?3|B\^D|:\'\-?\)|>:\[|:\-?\(|:\-?c|:\-?<|:\-?\[|:\{|;\(|:\-\|\||:@|>:\(|:\'\-?\(|D:<?|D[8;=X]|v.v|D\-\':|>:[\/]|:\-[./]|:[\/LS]|=[\/L]|>.<|:\$|>:\-?\)|>;\)|\}:\-?\)|3:\-?\)|\(>_<\)>?|^_?^;|\(#\^.\^#\)|[Oo]_[Oo]|:\-?o',s))

def getNumberOfElongatedWords(s):
    return len(re.findall('([a-zA-Z])\\1{2,}', s))
    
def pronoun(sen):
	
	for x in sen:
		if PRONOUN.__contains__(x)==True:
			return 1
	return 0

def exclamation(s):
	c = len(re.findall(r"[!]", s))
	if c>=1:
		return 1
	return 0

def question(s):
	return len(re.findall(r"[?]", s))

def intensifier(sen):

	count = 0
	for x in sen:
		if INTENSIFIER.__contains__(x)==True:
			count+=1
			#return 1
	if count>0:
		return 1
	return 0

def whword(sen):
	
	for x in sen:
		if WHWORD.__contains__(x)==True:
			return 1
	return 0

def religion(sen):
	
	for x in sen:
		if RELIGION.__contains__(x)==True:
			return 1
	return 0

def slang(sen):
	
	for x in sen:
		if SLANG.__contains__(x)==True:
			return 1
	return 0

def event_phrase(sen):
	
	for x in sen:
		if EVENT.__contains__(x)==True:
			return 1
	return 0

def modal(sen):
	for x in sen:
		if MODAL.__contains__(x)==True:
			return 1
	return 0

def subjectivity(sen):
	
        c = 0
        for x in sen:
                if SUBJECTIVE.__contains__(x)==True:
                        c+=1
        tot = len(sen) + 4.0 - 4.0 - 1.0
        num = c + 4.0 - 4.0
	try:
        	s = round(num/tot,4)
	except:
		s = 0
	if c>0:
		return c
	return 0
            
if __name__ == '__main__':

	READ_FILES()
	tok = Tokenizer(preserve_case=False)
	TAGREJECT = ['@','#','~','U','E']

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll ' + sys.argv[1] + ' > tagfile.txt'
	os.system(command)
	
	fp = open('tagfile.txt','r')
	fs = open(sys.argv[1],'r')
	s = ''
	N = 0
	feature = []
	for l in fp:
		wl = l.split('\t')
		if len(wl)>1:
			word = wl[0].strip(' \t\n\r').lower()
			tag = wl[1].strip(' \t\n\r')
			if tag=='N':
				try:
					w = lmtzr.lemmatize(word)
					word = w
				except Exception as e:
					pass
			elif tag=='V':
				try:
                                	w = Word(word)
                                	x = w.lemmatize("v")
					word = x
				except Exception as e:
                                	pass
			elif tag=='$':
				N+=1
			else:
				pass
			try:
				if tag not in TAGREJECT:
					s = s + word + ' '
			except Exception as e:
				pass
		else:
			unigram = tok.tokenize(s)
			bigram = []
			trigram = []
			if len(unigram)>1:
				for i in range(0,len(unigram)-1,1):
					s = unigram[i] + ' ' + unigram[i+1]
					bigram.append(s)
			if len(unigram)>2:
				for i in range(0,len(unigram)-2,1):
					s = unigram[i] + ' ' + unigram[i+1] + ' ' + unigram[i+2]
					trigram.append(s)
			Ngram = unigram + bigram + trigram
			row = fs.readline().strip(' \t\n\r')
			temp = tok.tokenize(row)
			New = []
			if len(temp)>1:
				for i in range(0,len(temp)-1,1):
					s = temp[i] + ' ' + temp[i+1]
					New.append(s)
			if len(temp)>2:
				for i in range(0,len(temp)-2,1):
					s = temp[i] + ' ' + temp[i+1] + ' ' + temp[i+2]
					New.append(s)
			
			Ngram = Ngram + temp + New
			Ngram = set(Ngram)
			Ngram = list(Ngram)
			E = exclamation(row)
			Q = question(row)
			M = modal(Ngram)
			I = intensifier(Ngram)
			W = whword(Ngram)
			EP = event_phrase(Ngram)
			S = subjectivity(temp)
			SG = slang(Ngram)
			P = pronoun(Ngram)
			RL = religion(Ngram)
			t = [N,E,Q,M,I,W,S,P,EP,SG,RL]
			feature.append(t)
			N = 0
			s = ''
	fp.close()
	fs.close()

	fo = open(sys.argv[2],'w')
	for x in feature:
		s = ''
		for i in range(0,len(x),1):
			s = s + str(x[i]) + '\t'
		fo.write(s.strip('\t') + '\n')
	fo.close()
