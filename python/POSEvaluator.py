import os
import re
from sets import Set
import nltk
import spacy
from spacy.tokens import Doc
from spacy.pipeline import Tagger
from textblob._text import (Parser as _Parser, Lexicon, WORD, POS, CHUNK, PNP, PENN, UNIVERSAL, Spelling)
from textblob._text import find_tags

TestData = "../data/00-23.br"

def readData():
	data_stream = open(TestData, "r")
	raw = data_stream.readlines()
	raw = [x.strip() for x in raw]
	ret = list()
	for line in raw:
		pairs = re.findall('\((.*?)\)', line)
		posTags = list()
		tokens = list()
		for pair in pairs:
			posTag = pair.split(" ")[0]
			token = pair.split(" ")[1]
			posTags.append(posTag)
			tokens.append(token)
		ret.append((posTags, tokens))
	return ret
		
def NLTK():
	data = readData()
	predictMap = {}
	goldMap = {}
	correctMap = {}
	for sentence in data:
		posTags = sentence[0]
		tokens = sentence[1]	
		tagged = nltk.pos_tag(tokens)
		for i in range (0, len(posTags)):
			goldTag = posTags[i]
			predictTag = tagged[i][1]
			if (predictTag == "("):
				predictTag = "-LRB-"
			if (predictTag == ")"):
				predictTag = "-RRB-"
			incrementMap(predictMap, predictTag)
			incrementMap(goldMap, goldTag)
			if (goldTag == predictTag):
				incrementMap(correctMap, goldTag)
	performance = producePerformance(goldMap, predictMap, correctMap)
	print "==========NLTK POS PERFORMANCE=========="
	printPerformance(performance)

def Spacy():
	data = readData()
	predictMap = {}
	goldMap = {}
	correctMap = {}
	nlp = spacy.load('en')
	tagger = Tagger(nlp.vocab)
	for sentence in data:
		posTags = sentence[0]
		tokens = [x.decode('utf-8') for x in sentence[1]]	
		doc = Doc(nlp.vocab, words=tokens)
		for name, proc in nlp.pipeline:
			if (name != "tagger"):
				continue
			doc = proc(doc)
			for i in range(0, len(doc)):
				predictTag = doc[i].tag_
				goldTag = posTags[i]
				incrementMap(predictMap, predictTag)
				incrementMap(goldMap, goldTag)
				if (goldTag == predictTag):
					incrementMap(correctMap, goldTag)
	performance = producePerformance(goldMap, predictMap, correctMap)
	print "==========Spacy POS PERFORMANCE=========="
	printPerformance(performance)

class Parser(_Parser):    
	def find_lemmata(self, tokens, **kwargs):
		return find_lemmata(tokens)
	def find_tags(self, tokens, **kwargs):
		if kwargs.get("tagset") in (PENN, None):
			kwargs.setdefault("map", lambda token, tag: (token, tag))
		if kwargs.get("tagset") == UNIVERSAL:
			kwargs.setdefault("map", lambda token, tag: penntreebank2universal(token, tag))
		return _Parser.find_tags(self, tokens, **kwargs)

MODULE = "../TextBlobModels/"

lexicon = Lexicon(
	path = MODULE + "en-lexicon.txt",
	morphology = MODULE + "en-morphology.txt",
	context = MODULE + "en-context.txt",
	entities = MODULE + "en-entities.txt",
	language = "en"
)
parser = Parser(
	lexicon = lexicon,
	default = ("NN", "NNP", "CD"),
	language = "en"
)

def TextBlob_PatternTagger():
	data = readData()
	predictMap = {}
	goldMap = {}
	correctMap = {}
	for sentence in data:
		posTags = sentence[0]
		tokens = sentence[1]
		outputs = parser.find_tags(tokens)
		for i in range (0, len(outputs)):
			goldTag = posTags[i]
			predictTag = outputs[i][1].encode('utf-8')
			if (predictTag == "("):
				predictTag = "-LRB-"
			if (predictTag == ")"):
				predictTag = "-RRB-"
			incrementMap(predictMap, predictTag)
			incrementMap(goldMap, goldTag)
			if (goldTag == predictTag):
				incrementMap(correctMap, goldTag)
	performance = producePerformance(goldMap, predictMap, correctMap)
	print "==========TextBlob Pattern POS PERFORMANCE=========="
	printPerformance(performance)
			
def incrementMap(m, k):
	if k in m:
		m[k] = m[k] + 1
	else:
		m[k] = 1

def producePerformance(gMap, pMap, cMap):
	labelSet = Set()
	ret = {}
	totalLabeled = 0
	totalCorrect = 0
	for gKey in gMap:
		labelSet.add(gKey)
	for pKey in pMap:
		labelSet.add(pKey)
	for label in labelSet:
		labeled = 0
		if (label in gMap):
			labeled = gMap[label]
		predicted = 0
		if (label in pMap):
			predicted = pMap[label]
		correct = 0
		if (label in cMap):
			correct = cMap[label]
		precision = 0.0
		if (predicted != 0):
			precision = float(correct) / float(predicted)
		recall =  0.0
		if (labeled != 0):
			recall = float(correct) / float(labeled)
		ret[label] = (precision, recall)
		totalLabeled += labeled
		totalCorrect += correct
	accuracy = float(totalCorrect) / float(totalLabeled)
	ret["ACC"] = (accuracy, accuracy)
	return ret

def printPerformance(resultMap):
	labelList = resultMap.keys()
	labelList.sort()
	accF1 = 0.0
	for label in labelList:
		precision = resultMap[label][0]
		recall = resultMap[label][1]
		f1 = 0.0
		if (precision * recall != 0):
			f1 = 2 * precision * recall / (precision + recall)
		accF1 += f1
		pStr = str(round(precision * 100.0, 2))
		rStr = str(round(recall * 100.0, 2))
		fStr = str(round(f1 * 100.0, 2))
		print label + "\t" + pStr + "\t" + rStr + "\t" + fStr
	print "Average F1\t" + str(round(100.0 * accF1 / float(len(labelList)), 2))
	print "Accuracy\t" + str(round(100.0 * resultMap["ACC"][0], 2))
			
def runTests():
	NLTK()
	Spacy()
	TextBlob_PatternTagger()

runTests()
