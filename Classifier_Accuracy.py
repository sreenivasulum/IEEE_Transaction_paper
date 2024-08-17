'''
Created on 11-May-2015

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
from sklearn import svm
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import cross_validation
import gzip
from happyfuntokenizing import *
import numpy as np
from sklearn.naive_bayes import *
#from happyfuntokenizing import *

lmtzr = WordNetLemmatizer()
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)

PRONOUN_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_pronoun.txt'
WHWORD_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_whwords.txt'
SLANG_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_swear.txt'
INTENSIFIER_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_intensifier.txt'
SUBJECTIVE_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/subjclueslen1-HLTEMNLP05.tff'
EVENT_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_nonsituational_phrase.txt'
MODAL_VERB_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/english_modal_verb.txt'
RELIGION_PATH = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Lexical_resources/classifier_dictionary/communal_race.txt'
#NONSIT_PATH = 'Common_nonsituational_word.txt'
#OPINION_HASHTAG_PATH = '/home/krudra/twitter_code/shared/language_overlap/code_mix_pitch/devanagari/devanagari_hashtag_opinion.txt'
#MENTION_PATH = '/home/krudra/twitter_code/shared/language_overlap/code_mix_pitch/devanagari/news_mention.txt'

#TRAIN1 = '../classification_4.3/fragmented_train_data/hydb_fragment_train.txt'
#TRAIN2 = '../classification_4.3/fragmented_train_data/utkd_fragment_train.txt'
#TRAIN3 = '../classification_4.3/fragmented_train_data/sandy_hook_fragment_train.txt'
#TRAIN4 = '../classification_4.3/fragmented_train_data/hagupit_fragment_train.txt'

#TRAIN1 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/fragmented_train_data/hydb_fragment_train.txt'
#TRAIN2 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/fragmented_train_data/utkd_fragment_train.txt'
#TRAIN3 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/fragmented_train_data/sandy_hook_fragment_train.txt'
#TRAIN4 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/fragmented_train_data/hagupit_fragment_train.txt'

TRAIN1 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/RAW_TRAIN DATA/hagupit_RAW_TWEET.txt'
TRAIN2 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/RAW_TRAIN DATA/hydb_RAW_TWEET.txt'
TRAIN3 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/RAW_TRAIN DATA/sandy_hook_RAW_TWEET.txt'
TRAIN4 = '/home/sreenu/Desktop/rudra/koustav_phdthesis_2018-master/Chapter_3/classification/RAW_TRAIN DATA/utkd_RAW_TWEET.txt'



#TRAIN1="../dataset4may/SMERP/RetrievedSMERP/level-1/TotalSMERP_T31.tsv"

'''TRAIN1 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hydb_RAW_CLASS.txt'
TRAIN2 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/utkd_RAW_CLASS.txt'
TRAIN3 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/sandy_hook_RAW_CLASS.txt'
TRAIN4 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hagupit_RAW_CLASS.txt' '''

'''TRAIN1 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hydb_balance_RAW.txt'
TRAIN2 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/utkd_balance_RAW.txt'
TRAIN3 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/sandy_hook_balance_RAW.txt'
TRAIN4 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hagupit_balance_RAW.txt' '''


emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

TAGGER_PATH = '../../Lexical_resources/ark-tweet-nlp-0.3.2/'

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

	'''fp = open(NONSIT_PATH,'r')
        for l in fp:
		EVENT[l.strip(' \t\n\r').lower()] = 1
        fp.close() '''
	
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

def getHashtagopinion(sen):
	fp = codecs.open(OPINION_HASHTAG_PATH,'r','utf-8')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_hash = set([])
	for x in sen:
		if x.startswith('#')==True:
			cur_hash.add(x.strip(' \t\n\r').lower())
	size = len(temp.intersection(cur_hash))
	if size>0:
		return 1
	return 0

def numeral(temp):
	c = 0
	for x in temp:
		if x.isdigit()==True:
			c+=1
	return c

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
	#print(s)
	#return s

#HashTagSentimentUnigrams("Oh wrestle", "NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt")
#HashTagSentimentUnigrams("Oh no wrestle", "Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt")
#print negatedContextCount("this is not for me")
            
if __name__ == '__main__':

	READ_FILES()
	tok = Tokenizer(preserve_case=False)

	fo = open('temp.txt','w')
	fp = open(TRAIN1,'r')
	for l in fp:
	     wl = l.split('\t')
	     fo.write(wl[0].strip(' \t\n\r') + '\n')
	fp.close()
	fo.close()

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll temp.txt > tagfile.txt'
	os.system(command)
	
	fp = open('tagfile.txt','r')
	fs = open(TRAIN1,'r')
	s = ''
	N = 0
	hydb_feature = []
	hydb_label = []
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
			row = fs.readline().split('\t')
			temp = tok.tokenize(row[0])
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
			print(unigram)
			print(temp)
			print(Ngram)
			sys.exit(0)
			E = exclamation(row[0])
			Q = question(row[0])
			M = modal(Ngram)
			I = intensifier(Ngram)
			W = whword(Ngram)
			EP = event_phrase(Ngram)
			S = subjectivity(temp)
			SG = slang(Ngram)
			P = pronoun(Ngram)
			EL = getNumberOfElongatedWords(row[0])
			RL = religion(Ngram)
			#EM = emoticons(org_tweet[row[1].strip(' \t\n\r')])
			#SM = smileys(row[0].strip(' \t\n\r'))
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EM,SM]
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EL]
			t = [N,E,Q,M,I,W,S,P,EP,SG,RL]
			hydb_feature.append(t)
			hydb_label.append(int(row[1]))
			N = 0
			s = ''
	fp.close()
	fs.close()
	
	IN = []
	CR = []
	PR = []
	RC = []
	FS = []

        hydb_clf = svm.SVC(kernel='rbf',gamma=0.5)
        #hydb_clf = svm.SVC(kernel='rbf')
        #hydb_clf = RandomForestClassifier()
        #hydb_clf = svm.LinearSVC()
        #hydb_clf = LogisticRegression()
        #hydb_clf = BernoulliNB()
        hydb_clf.fit(hydb_feature,hydb_label)
        scores = cross_validation.cross_val_score(hydb_clf,hydb_feature,hydb_label,cv=10)
        print('Hydb: ',scores.mean(),scores.std())
	recall = cross_validation.cross_val_score(hydb_clf, hydb_feature, hydb_label, cv=10, scoring='recall')
        print('Recall', np.mean(recall), recall)
        precision = cross_validation.cross_val_score(hydb_clf, hydb_feature, hydb_label, cv=10, scoring='precision')
        print('Precision', np.mean(precision), precision)
        f1 = cross_validation.cross_val_score(hydb_clf, hydb_feature, hydb_label, cv=10, scoring='f1')
        print('F1', np.mean(f1), f1)
	IN.append(scores.mean())

	fo = open('temp.txt','w')
	fp = open(TRAIN2,'r')
	for l in fp:
		wl = l.split('\t')
		fo.write(wl[0].strip(' \t\n\r') + '\n')
	fp.close()
	fo.close()

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll temp.txt > tagfile.txt'
	os.system(command)
	
	fp = open('tagfile.txt','r')
	fs = open(TRAIN2,'r')
	s = ''
	N = 0
	utkd_feature = []
	utkd_label = []
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
			row = fs.readline().split('\t')
			temp = tok.tokenize(row[0])
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
			E = exclamation(row[0])
			Q = question(row[0])
			M = modal(Ngram)
			I = intensifier(Ngram)
			W = whword(Ngram)
			EP = event_phrase(Ngram)
			S = subjectivity(temp)
			SG = slang(Ngram)
			P = pronoun(Ngram)
			EL = getNumberOfElongatedWords(row[0])
			RL = religion(Ngram)
			#EM = emoticons(org_tweet[row[1].strip(' \t\n\r')])
			#SM = smileys(row[0].strip(' \t\n\r'))
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EM,SM]
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EL]
			t = [N,E,Q,M,I,W,S,P,EP,SG,RL]
			utkd_feature.append(t)
			utkd_label.append(int(row[1]))
			N = 0
			s = ''
	fp.close()
	fs.close()
        
	utkd_clf = svm.SVC(kernel='rbf',gamma=0.5)
        #utkd_clf = svm.SVC(kernel='rbf')
        #utkd_clf = RandomForestClassifier()
        #utkd_clf = svm.LinearSVC()
        #utkd_clf = LogisticRegression()
        #utkd_clf = BernoulliNB()
        utkd_clf.fit(utkd_feature,utkd_label)
        scores = cross_validation.cross_val_score(utkd_clf,utkd_feature,utkd_label,cv=10)
        print('Utkd: ',scores.mean(),scores.std())
	recall = cross_validation.cross_val_score(utkd_clf, utkd_feature, utkd_label, cv=10, scoring='recall')
        print('Recall', np.mean(recall), recall)
        precision = cross_validation.cross_val_score(utkd_clf, utkd_feature, utkd_label, cv=10, scoring='precision')
        print('Precision', np.mean(precision), precision)
        f1 = cross_validation.cross_val_score(utkd_clf, utkd_feature, utkd_label, cv=10, scoring='f1')
        print('F1', np.mean(f1), f1)
	IN.append(scores.mean())
	

	fo = open('temp.txt','w')
	fp = open(TRAIN3,'r')
	for l in fp:
		wl = l.split('\t')
		fo.write(wl[0].strip(' \t\n\r') + '\n')
	fp.close()
	fo.close()

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll temp.txt > tagfile.txt'
	os.system(command)
	
	fp = open('tagfile.txt','r')
	fs = open(TRAIN3,'r')
	s = ''
	N = 0
	sandy_hook_feature = []
	sandy_hook_label = []
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
			row = fs.readline().split('\t')
			temp = tok.tokenize(row[0])
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
			E = exclamation(row[0])
			Q = question(row[0])
			M = modal(Ngram)
			I = intensifier(Ngram)
			W = whword(Ngram)
			EP = event_phrase(Ngram)
			S = subjectivity(temp)
			SG = slang(Ngram)
			P = pronoun(Ngram)
			EL = getNumberOfElongatedWords(row[0])
			RL = religion(Ngram)
			#EM = emoticons(org_tweet[row[1].strip(' \t\n\r')])
			#SM = smileys(row[0].strip(' \t\n\r'))
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EM,SM]
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EL]
			t = [N,E,Q,M,I,W,S,P,EP,SG,RL]
			sandy_hook_feature.append(t)
			sandy_hook_label.append(int(row[1]))
			N = 0
			s = ''
	fp.close()
	fs.close()

        sandy_hook_clf = svm.SVC(kernel='rbf',gamma=0.5)
        #sandy_hook_clf = svm.SVC(kernel='rbf')
        #sandy_hook_clf = RandomForestClassifier()
        #sandy_hook_clf = svm.LinearSVC()
        #sandy_hook_clf = LogisticRegression()
        #sandy_hook_clf = BernoulliNB()
        sandy_hook_clf.fit(sandy_hook_feature,sandy_hook_label)
        scores = cross_validation.cross_val_score(sandy_hook_clf,sandy_hook_feature,sandy_hook_label,cv=10)
        print('SandyHook: ',scores.mean(),scores.std())
	recall = cross_validation.cross_val_score(sandy_hook_clf, sandy_hook_feature, sandy_hook_label, cv=10, scoring='recall')
        print('Recall', np.mean(recall), recall)
        precision = cross_validation.cross_val_score(sandy_hook_clf, sandy_hook_feature, sandy_hook_label, cv=10, scoring='precision')
        print('Precision', np.mean(precision), precision)
        f1 = cross_validation.cross_val_score(sandy_hook_clf, sandy_hook_feature, sandy_hook_label, cv=10, scoring='f1')
        print('F1', np.mean(f1), f1)
	IN.append(scores.mean())
	

	fo = open('temp.txt','w')
	fp = open(TRAIN4,'r')
	for l in fp:
		wl = l.split('\t')
		fo.write(wl[0].strip(' \t\n\r') + '\n')
	fp.close()
	fo.close()

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll temp.txt > tagfile.txt'
	os.system(command)
	
	fp = open('tagfile.txt','r')
	fs = open(TRAIN4,'r')
	s = ''
	N = 0
	hagupit_feature = []
	hagupit_label = []
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
			row = fs.readline().split('\t')
			temp = tok.tokenize(row[0])
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
			E = exclamation(row[0])
			Q = question(row[0])
			M = modal(Ngram)
			I = intensifier(Ngram)
			W = whword(Ngram)
			EP = event_phrase(Ngram)
			S = subjectivity(temp)
			SG = slang(Ngram)
			P = pronoun(Ngram)
			EL = getNumberOfElongatedWords(row[0])
			RL = religion(Ngram)
			#EM = emoticons(org_tweet[row[1].strip(' \t\n\r')])
			#SM = smileys(row[0].strip(' \t\n\r'))
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EM,SM]
			#t = [N,E,Q,M,I,W,S,P,EP,SG,EL]
			t = [N,E,Q,M,I,W,S,P,EP,SG,RL]
			hagupit_feature.append(t)
			hagupit_label.append(int(row[1]))
			N = 0
			s = ''
	fp.close()
	fs.close()

        hagupit_clf = svm.SVC(kernel='rbf',gamma=0.5)
        #hagupit_clf = svm.SVC(kernel='rbf')
        #hagupit_clf = RandomForestClassifier()
        #hagupit_clf = svm.LinearSVC()
        #hagupit_clf = LogisticRegression()
        #hagupit_clf = BernoulliNB()
        hagupit_clf.fit(hagupit_feature,hagupit_label)
        scores = cross_validation.cross_val_score(hagupit_clf,hagupit_feature,hagupit_label,cv=10)
        print('Hagupit: ',scores.mean(),scores.std())
	recall = cross_validation.cross_val_score(hagupit_clf, hagupit_feature, hagupit_label, cv=10, scoring='recall')
        print('Recall', np.mean(recall), recall)
        precision = cross_validation.cross_val_score(hagupit_clf, hagupit_feature, hagupit_label, cv=10, scoring='precision')
        print('Precision', np.mean(precision), precision)
        f1 = cross_validation.cross_val_score(hagupit_clf, hagupit_feature, hagupit_label, cv=10, scoring='f1')
        print('F1', np.mean(f1), f1)
	IN.append(scores.mean())
			
        
	print('Train Hydb')
        print('Utkd: ',hydb_clf.score(utkd_feature,utkd_label))
        print('Sandy_Hook: ',hydb_clf.score(sandy_hook_feature,sandy_hook_label))
        print('Hagupit: ',hydb_clf.score(hagupit_feature,hagupit_label))
	CR.append(hydb_clf.score(utkd_feature,utkd_label))
	CR.append(hydb_clf.score(sandy_hook_feature,sandy_hook_label))
	CR.append(hydb_clf.score(hagupit_feature,hagupit_label))

        print('Train Utkd')
        print('Hydb: ',utkd_clf.score(hydb_feature,hydb_label))
        print('Sandy_Hook: ',utkd_clf.score(sandy_hook_feature,sandy_hook_label))
        print('Hagupit: ',utkd_clf.score(hagupit_feature,hagupit_label))
	CR.append(utkd_clf.score(hydb_feature,hydb_label))
	CR.append(utkd_clf.score(sandy_hook_feature,sandy_hook_label))
	CR.append(utkd_clf.score(hagupit_feature,hagupit_label))

        print('Train Sandy_Hook')
        print('Hydb: ',sandy_hook_clf.score(hydb_feature,hydb_label))
        print('Utkd: ',sandy_hook_clf.score(utkd_feature,utkd_label))
        print('Hagupit: ',sandy_hook_clf.score(hagupit_feature,hagupit_label))
	CR.append(sandy_hook_clf.score(hydb_feature,hydb_label))
	CR.append(sandy_hook_clf.score(utkd_feature,utkd_label))
	CR.append(sandy_hook_clf.score(hagupit_feature,hagupit_label))

        print('Train Hagupit')
        print('Hydb: ',hagupit_clf.score(hydb_feature,hydb_label))
        print('Utkd: ',hagupit_clf.score(utkd_feature,utkd_label))
        print('Sandy_Hook: ',hagupit_clf.score(sandy_hook_feature,sandy_hook_label))
	CR.append(hagupit_clf.score(hydb_feature,hydb_label))
	CR.append(hagupit_clf.score(utkd_feature,utkd_label))
	CR.append(hagupit_clf.score(sandy_hook_feature,sandy_hook_label))
	print('\n\n')
	
	########################################## Precision Recall F-score ###################################################
	utkd_predicted_label = hydb_clf.predict(utkd_feature)
	print('Hydb -> utkd | sandy_hook | hagupit')
	print('PRECISION: ',metrics.precision_score(utkd_label,utkd_predicted_label))
        print('RECALL: ',metrics.recall_score(utkd_label,utkd_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(utkd_label,utkd_predicted_label))
	PR.append(metrics.precision_score(utkd_label,utkd_predicted_label))
	RC.append(metrics.recall_score(utkd_label,utkd_predicted_label))
	FS.append(metrics.f1_score(utkd_label,utkd_predicted_label))
	
	sandy_hook_predicted_label = hydb_clf.predict(sandy_hook_feature)
	print('PRECISION: ',metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        print('RECALL: ',metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	PR.append(metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        RC.append(metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        FS.append(metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	
	hagupit_predicted_label = hydb_clf.predict(hagupit_feature)
	print('PRECISION: ',metrics.precision_score(hagupit_label,hagupit_predicted_label))
        print('RECALL: ',metrics.recall_score(hagupit_label,hagupit_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hagupit_label,hagupit_predicted_label))
	PR.append(metrics.precision_score(hagupit_label,hagupit_predicted_label))
        RC.append(metrics.recall_score(hagupit_label,hagupit_predicted_label))
        FS.append(metrics.f1_score(hagupit_label,hagupit_predicted_label))

	print('\n\n')
	
	#########################################################################################################################
	hydb_predicted_label = utkd_clf.predict(hydb_feature)
	print('Utkd -> hydb | sandy_hook | hagupit')
	print('PRECISION: ',metrics.precision_score(hydb_label,hydb_predicted_label))
        print('RECALL: ',metrics.recall_score(hydb_label,hydb_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hydb_label,hydb_predicted_label))
	PR.append(metrics.precision_score(hydb_label,hydb_predicted_label))
        RC.append(metrics.recall_score(hydb_label,hydb_predicted_label))
        FS.append(metrics.f1_score(hydb_label,hydb_predicted_label))
	
	sandy_hook_predicted_label = utkd_clf.predict(sandy_hook_feature)
	print('PRECISION: ',metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        print('RECALL: ',metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	PR.append(metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        RC.append(metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        FS.append(metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	
	hagupit_predicted_label = utkd_clf.predict(hagupit_feature)
	print('PRECISION: ',metrics.precision_score(hagupit_label,hagupit_predicted_label))
        print('RECALL: ',metrics.recall_score(hagupit_label,hagupit_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hagupit_label,hagupit_predicted_label))
	PR.append(metrics.precision_score(hagupit_label,hagupit_predicted_label))
        RC.append(metrics.recall_score(hagupit_label,hagupit_predicted_label))
        FS.append(metrics.f1_score(hagupit_label,hagupit_predicted_label))
	print('\n\n')

	###########################################################################################################################
	hydb_predicted_label = sandy_hook_clf.predict(hydb_feature)
	print('Sandy_hook -> hydb | utkd | hagupit')
	print('PRECISION: ',metrics.precision_score(hydb_label,hydb_predicted_label))
        print('RECALL: ',metrics.recall_score(hydb_label,hydb_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hydb_label,hydb_predicted_label))
	PR.append(metrics.precision_score(hydb_label,hydb_predicted_label))
        RC.append(metrics.recall_score(hydb_label,hydb_predicted_label))
        FS.append(metrics.f1_score(hydb_label,hydb_predicted_label))
	
	utkd_predicted_label = sandy_hook_clf.predict(utkd_feature)
	print('PRECISION: ',metrics.precision_score(utkd_label,utkd_predicted_label))
        print('RECALL: ',metrics.recall_score(utkd_label,utkd_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(utkd_label,utkd_predicted_label))
	PR.append(metrics.precision_score(utkd_label,utkd_predicted_label))
	RC.append(metrics.recall_score(utkd_label,utkd_predicted_label))
	FS.append(metrics.f1_score(utkd_label,utkd_predicted_label))
	
	hagupit_predicted_label = sandy_hook_clf.predict(hagupit_feature)
	print('PRECISION: ',metrics.precision_score(hagupit_label,hagupit_predicted_label))
        print('RECALL: ',metrics.recall_score(hagupit_label,hagupit_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hagupit_label,hagupit_predicted_label))
	PR.append(metrics.precision_score(hagupit_label,hagupit_predicted_label))
        RC.append(metrics.recall_score(hagupit_label,hagupit_predicted_label))
        FS.append(metrics.f1_score(hagupit_label,hagupit_predicted_label))
	print('\n\n')

	############################################################################################################################
	
	print('Hagupit -> hydb | utkd | sandy_hook')
	hydb_predicted_label = hagupit_clf.predict(hydb_feature)
	print('PRECISION: ',metrics.precision_score(hydb_label,hydb_predicted_label))
        print('RECALL: ',metrics.recall_score(hydb_label,hydb_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(hydb_label,hydb_predicted_label))
	PR.append(metrics.precision_score(hydb_label,hydb_predicted_label))
        RC.append(metrics.recall_score(hydb_label,hydb_predicted_label))
        FS.append(metrics.f1_score(hydb_label,hydb_predicted_label))
	
	utkd_predicted_label = hagupit_clf.predict(utkd_feature)
	print('PRECISION: ',metrics.precision_score(utkd_label,utkd_predicted_label))
        print('RECALL: ',metrics.recall_score(utkd_label,utkd_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(utkd_label,utkd_predicted_label))
	PR.append(metrics.precision_score(utkd_label,utkd_predicted_label))
	RC.append(metrics.recall_score(utkd_label,utkd_predicted_label))
	FS.append(metrics.f1_score(utkd_label,utkd_predicted_label))
	
	sandy_hook_predicted_label = hagupit_clf.predict(sandy_hook_feature)
	print('PRECISION: ',metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        print('RECALL: ',metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        print('F1_SCORE: ',metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	PR.append(metrics.precision_score(sandy_hook_label,sandy_hook_predicted_label))
        RC.append(metrics.recall_score(sandy_hook_label,sandy_hook_predicted_label))
        FS.append(metrics.f1_score(sandy_hook_label,sandy_hook_predicted_label))
	
	print(len(IN),len(CR),len(PR),len(RC),len(FS))
        print('Average Indomain Accuracy: ',np.mean(IN))
        print('Average Cross-domain Accuracy: ',np.mean(CR))
        print('Average and std Precision: ',np.mean(PR),np.std(PR))
        print('Average and std Recall: ',np.mean(RC),np.std(RC))
        print('Average and std F-score: ',np.mean(FS),np.std(FS))
