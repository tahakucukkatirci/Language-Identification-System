# -*- coding: utf-8 -*-
import codecs
import sys
import os
from random import shuffle
from math import inf,log
from collections import Counter
from time import time
from itertools import permutations

def split_sentence(sentence):
	temp = sentence.decode("utf-8")
	splitted = temp.split("\t")

	return splitted

def split_train_and_test(fileName):
	with open(fileName, 'rb') as f:
	  contents = f.readlines()

	training_set = []
	test_set = []
	for i in range(13):
		start=i*2000
		end  =start+2000
		current = contents[start:end]
		shuffle(current)
		splitter = int(len(current)*0.9)
		training_set = training_set + current[:splitter]
		test_set = test_set + current[splitter:]
	

	return training_set,test_set

def train(train_set):
	lang_char_dict = dict()
	for sentence in train_set:
		splitted = split_sentence(sentence)
		language = splitted[1][:-2]
		chars    = [l for l in splitted[0] if l!=' ']
		if language in lang_char_dict.keys():
			lang_char_dict[language] += chars
		else:
			lang_char_dict[language] = chars

	for key in lang_char_dict:
		lang_char_dict[key] = Counter(lang_char_dict[key])

	return lang_char_dict

def split_test_set(test_set):
	test_X=[]
	test_y=[]

	for sentence in test_set:
		temp = split_sentence(sentence)
		test_X.append(temp[0])
		test_y.append(temp[1][:-2])

	return test_X, test_y

def P_char_given_lang(chars,lang,train_dict):
	prob=0
	V = len(train_dict[lang].keys())
	N = sum(train_dict[lang].values())
	for char in chars:
		prob += log(float((train_dict[lang][char] + 1))/(N+V))
	
	return prob

def test(test_X,train_dict):
	pred_y=[]
	languages = train_dict.keys()
	for test_sentence in test_X:
		chars = [l for l in test_sentence if l!=' ']
		prediction=''
		max_prob = -inf
		for language in languages:
			temp_prob = P_char_given_lang(chars,language,train_dict)
			if(temp_prob>max_prob):
				max_prob = temp_prob
				prediction = language

		pred_y.append(prediction)

	return pred_y

def accuracy(test_y,pred_y,language=' '):
	if language==' ':
		no_of_corrects = len(["." for i in range(len(test_y)) if test_y[i]==pred_y[i]])
		return float(no_of_corrects)/len(test_y)

	else:
		indices = [i for i, l in enumerate(test_y) if l == language]
		no_of_corrects = len(["." for index in indices if test_y[index]==pred_y[index]])
		return float(no_of_corrects)/len(indices)
	
	return 0.0

def precision_recall_f(languages,test_y,pred_y,choice='micro'):
	TP_list = []
	FN_list = []
	FP_list = []
	precision = 0
	recall = 0
	f_score = 0
	for language in languages:
		TP = 0
		TN = 0
		FP = 0
		FN = 0
		for i in range(len(test_y)):
			if test_y[i]==language and pred_y[i]==language:
				TP+=1
			elif test_y[i]==language and pred_y[i]!=language:
				FN+=1
			elif test_y[i]!=language and pred_y[i]==language:
				FP+=1
			elif test_y[i]!=language and pred_y[i]!=language:
				TN+=1

		#print(TP," ",FN," ",FP," ",TN," ",(TP+FN+FP+TN))
		TP_list.append(float(TP))
		FN_list.append(float(FN))
		FP_list.append(float(FP))


	if choice=='micro':
		TP_sum = sum(TP_list)
		FN_sum = sum(FN_list)
		FP_sum = sum(FP_list)
		#print(TP_sum)
		#print(FN_sum)
		#print(FP_sum)
		precision = TP_sum/(TP_sum+FP_sum)
		recall    = TP_sum/(TP_sum+FN_sum)
		f_score   = (2*precision*recall)/(precision+recall)

	elif choice=='macro':
		precision_list = []
		recall_list = []
		f_score_list = []
		M = len(TP_list)
		for i in range(M):
			tp = TP_list[i]
			fp = FP_list[i]
			fn = FN_list[i]
			prec_i   = tp/(tp+fp) if tp+fp!=0 else 1
			recall_i = tp/(tp+fn) if tp+fn!=0 else 1
			f_score_i= (2*prec_i*recall_i)/(prec_i+recall_i) #if prec_i+recall_i!=0 else 0

			precision_list.append(prec_i)
			recall_list.append(recall_i)
			f_score_list.append(f_score_i)

		precision = sum(precision_list)/M
		recall    = sum(recall_list)/M
		f_score   = sum(f_score_list)/M

	return precision,recall,f_score

def svm_file_data(data_to_Write,languages,chars):
	lines=[]
	for sentence in data_to_Write:
		splitted = split_sentence(sentence)
		lang = splitted[1][:-2]
		line=''
		line += str(languages[lang]) + " "
		letters = [l for l in splitted[0] if l!=' ']
		letters = list(set(letters))
		letters = [chars[l] for l in letters]
		letters = sorted(letters)
		for l in letters:
			line+= str(l) + ":1 "

		#line = line.strip() + "\n"
		lines.append(line)

	return lines

def svm_bonus(data_to_Write,lines,chars):
	bigrams = permutations(chars.keys(), 2)
	bigrams = list(bigrams)
	for i in range(len(bigrams)):
		bigrams[i] = ''.join(map(str, bigrams[i]))

	for char in chars.keys():
		double = str(char)+str(char)
		bigrams.append(''.join(map(str, double)))

	bigrams = {bigram: index for index, bigram in enumerate(bigrams,start=len(chars)+1)}
	for i in range(len(data_to_Write)):
		splitted = split_sentence(data_to_Write[i])
		line=''
		letters = [l for l in splitted[0] if l!=' ']
		char_bigrams = [str(letters[j])+str(letters[j+1]) for j in range(len(letters)-1)]
		char_bigrams = list(set(char_bigrams))
		char_bigrams = [bigrams[c] for c in char_bigrams]
		char_bigrams = sorted(char_bigrams)
		for b in char_bigrams:
			line+= str(b) + ":1 "

		lines[i] = lines[i] + line

	return lines

def svm(bonus=False):
	train_set, test_set = split_train_and_test("corpus_new.txt")
	train_dict= train(train_set)
	test_dict = train(test_set)
	languages = list(train_dict.keys())
	chars = []
	for lang in languages:
		chars = chars + list(train_dict[lang].keys()) + list(test_dict[lang].keys())

	del train_dict
	del test_dict

	chars = list(set(chars))
	languages = {lang: index for index, lang in enumerate(languages, start=1)}
	chars 	  = {char: index for index, char in enumerate(chars,start=1)}
	
	lines = svm_file_data(train_set,languages,chars)
	if bonus:
		lines = svm_bonus(train_set,lines,chars)

	for i in range(len(lines)):
		lines[i] = lines[i].strip() + "\n"

	f = open('train.txt','w')
	f.writelines(lines)
	f.close()


	lines = svm_file_data(test_set,languages,chars)
	if bonus:
		lines = svm_bonus(test_set,lines,chars)

	for i in range(len(lines)):
		lines[i] = lines[i].strip() + "\n"
	f = open('test.txt','w')
	f.writelines(lines)
	f.close()

	return languages
	
def svm_output(lang_dict):
	if len(sys.argv)>2:
		languages = [str(i) for i in range(1,14)]
		test_y = []
		pred_y=[]

		f = open("test.txt","r")
		lines = f.readlines()
		f.close()
		test_y = [str(line.split(' ',1)[0]) for line in lines]

		f = open("output.txt","r")
		lines = f.readlines()
		f.close()
		pred_y = [str(line.split(' ',1)[0]) for line in lines]


		if "accuracy" in sys.argv:
			print("\n\n\n")
			print("Total accuracy =",accuracy(test_y,pred_y))
			for key in lang_dict.keys():
				if key in sys.argv:
					print(key,"accuracy =",accuracy(test_y,pred_y,str(lang_dict[key])))

		if "micro" in sys.argv:
			print("\n\n\n")
			p,r,f=precision_recall_f(languages,test_y,pred_y,choice="micro")
			print("Micro-averaged Precision =", p)
			print("Micro-averaged Recall =", r)
			print("Micro-averaged F-score =", f)

		if "macro" in sys.argv:
			print("\n\n\n")
			p,r,f=precision_recall_f(languages,test_y,pred_y,choice="macro")
			print("Macro-averaged Precision =", p)
			print("Macro-averaged Recall =", r)
			print("Macro-averaged F-score =", f)

def main():
	if "naiveBayes" in sys.argv:
		train_set, test_set = split_train_and_test("corpus_new.txt")
		train_dict = train(train_set)
		test_X, test_y = split_test_set(test_set)
		pred_y = test(test_X,train_dict)

		if "accuracy" in sys.argv:
			print("Total accuracy =",accuracy(test_y,pred_y))
			for key in train_dict.keys():
				if key in sys.argv:
					print(key,"accuracy =",accuracy(test_y,pred_y,key))


		if "micro" in sys.argv:
			print("\n\n\n")
			p,r,f=precision_recall_f(list(train_dict.keys()),test_y,pred_y,choice="micro")
			print("Micro-averaged Precision =", p)
			print("Micro-averaged Recall =", r)
			print("Micro-averaged F-score =", f)

		if "macro" in sys.argv:
			print("\n\n\n")
			p,r,f=precision_recall_f(list(train_dict.keys()),test_y,pred_y,choice="macro")
			print("Macro-averaged Precision =", p)
			print("Macro-averaged Recall =", r)
			print("Macro-averaged F-score =", f)

	elif "svm" in sys.argv:
		lang_dict = svm()
		os.system("./svm_stuff/svm_multiclass_learn -c 1.0 train.txt model.txt")
		os.system("./svm_stuff/svm_multiclass_classify test.txt model.txt output.txt")
		svm_output(lang_dict)


	elif "svm_bonus" in sys.argv:
		lang_dict = svm(bonus=True)
		os.system("./svm_stuff/svm_multiclass_learn -c 1.0 train.txt model.txt")
		os.system("./svm_stuff/svm_multiclass_classify test.txt model.txt output.txt")		
		svm_output(lang_dict)


if __name__ == "__main__":
    main()