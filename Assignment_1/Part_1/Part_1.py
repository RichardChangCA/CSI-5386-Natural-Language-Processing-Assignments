from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

# # in the first time you should use these two line codes
# import nltk
# nltk.download('punkt')

import nltk
from nltk.tokenize import word_tokenize
from spacy.symbols import ORTH, LEMMA, POS, TAG
from gensim.corpora import Dictionary
import re
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()


def tokenization_statistic(tokenization_all,Tokens_file_name,results_file_name):
    dct = Dictionary([tokenization_all])  # initialize a Dictionary
    # print(dct)
    unique_token_numbers = len(dct)
    # print("unique_token_numbers:",unique_token_numbers)

    word_frequency = []
    for word in tokenization_all:
        word_frequency.append(tokenization_all.count(word))
    num = 0
    dct_pairs = sorted(list(set(list(zip(tokenization_all,word_frequency)))),key=lambda x: x[1],reverse=True)
    with open(Tokens_file_name,'w') as frequency_file:
        frequency_file.write("number" + " " + "word" + " " + "frequency" +"\n")
        for item in dct_pairs:
            frequency_file.write(str(num) + " " + item[0] + " " + str(item[1]) +"\n")
            num += 1
    bag_of_words = dct.doc2bow(tokenization_all)
    # print(bag_of_words)
    token_numbers = 0
    for i in bag_of_words:
        token_numbers += i[1]
    # print("token_numbers:",token_numbers)

    ratio = unique_token_numbers / token_numbers

    # print("type/token ratio:", ratio)

    if(results_file_name == None):
        return
    
    with open(results_file_name, 'w') as f_ttr_results:
        f_ttr_results.write("unique_token_numbers:"+str(unique_token_numbers)+"\n")
        f_ttr_results.write("token_numbers:"+str(token_numbers)+"\n")
        f_ttr_results.write("type/token ratio:"+str(ratio)+"\n")
    
# encoding='utf-8-sig' is to remove \ufeff
# write the result into .txt file

# special case is sensitive
with open("transfered_special_case_2.txt","r") as f:
    for line in f:
        special_case = []
        line = line.strip('\n')
        splitted_line = line.split(" ")
        if('' in splitted_line):
            splitted_line.remove('')
        # print(splitted_line)
        for item_num in range(1,len(splitted_line)):
            special_case.append({ORTH:splitted_line[item_num]})
        nlp.tokenizer.add_special_case(splitted_line[0], special_case)

all_text =  open('twitter_corpus.txt', 'r', encoding='utf-8-sig')

f_tokenizer = open('microblog2011_tokenized.txt','w')
f_tokenizer_in_report = open('microblog2011_tokenized_report.txt','w')

num = 0

tokenization_all = []

for text_line in all_text:

    # tokenization_results = word_tokenize(text_line)
    text_line = text_line.strip('\n')
    text_line = re.sub(' +', ' ', text_line)
    tokenization_results = [w.text.lower() for w in nlp(text_line)]

    tokenization_all += tokenization_results

    f_tokenizer.write(str(tokenization_results)+'\n')

    if(num<20):
        f_tokenizer_in_report.write(str(num)+":\n")
        f_tokenizer_in_report.write(text_line)
        f_tokenizer_in_report.write(str(tokenization_results)+'\n')
    
    num += 1

f_tokenizer.close()
f_tokenizer_in_report.close()

tokenization_statistic(tokenization_all, 'Tokens.txt', 'type_token_ratio_results.txt')

stopwords_list = []
with open('stopwords.txt','r') as f:
    for line in f:
        stopwords_list.append(line[:-1])
# delete /n line break

punctuations="!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

punctuation_list = []
for punc in punctuations:
    punctuation_list.append(punc)

if(' ' in tokenization_all):
    tokenization_all.remove(' ')
if('...' in tokenization_all):
    tokenization_all.remove('...')
if('…' in tokenization_all):
    tokenization_all.remove('…')
if('..' in tokenization_all):
    tokenization_all.remove('..')
if('█' in tokenization_all):
    tokenization_all.remove('█')
 
filtered_sentence = []
for w in tokenization_all: 
    if w not in punctuation_list: 
        filtered_sentence.append(w)
# print(filtered_sentence)

tokenization_statistic(filtered_sentence, 'Tokens_without_punctuation.txt', None)



stopwords_with_punctuations = stopwords_list + punctuation_list

filtered_sentence = []
for w in tokenization_all: 
    if w not in stopwords_with_punctuations: 
        filtered_sentence.append(w)
# print(filtered_sentence)

tokenization_statistic(filtered_sentence, 'Tokens_without_stops.txt', 'type_token_ratio_without_stops_results.txt')

bigram_frequency = nltk.FreqDist(nltk.bigrams(filtered_sentence))

# print(bigram_frequency)

bigrams_list = []
for item in bigram_frequency.keys():
    bigrams_list.append((item,bigram_frequency[item]))
# print(bigrams_list)

bigrams_list = sorted(bigrams_list,key=lambda x: x[1],reverse=True)

num = 0
with open("bigrams.txt",'w') as bigrams_file:
    bigrams_file.write("number" + " " + "word" + " " + "frequency" +"\n")
    for item in bigrams_list:
        bigrams_file.write(str(num) + " " + item[0][0] +" "+ item[0][1] + " " + str(item[1]) +"\n")
        num += 1