from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

# # in the first time you should use these two line codes
# import nltk
# nltk.download('punkt')


from nltk.tokenize import word_tokenize
from spacy.symbols import ORTH, LEMMA, POS, TAG
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
# nlp = Tokenizer(nlp.vocab)
# nlp = nlp.Defaults.create_tokenizer(nlp)


# all_text =  open('twitter_corpus.txt', 'r')
# for text_line in all_text:
#     print(text_line)

#     #  "nlp" Object is used to create documents with linguistic annotations.
#     my_doc = nlp(text_line)

#     # Create list of word tokens
#     token_list = []
#     for token in my_doc:
#         token_list.append(token.text)
#     print(token_list)
    
# encoding='utf-8-sig' is to remove \ufeff
# write the result into .txt file
all_text =  open('twitter_corpus.txt', 'r', encoding='utf-8-sig')

f_tokenizer = open('microblog2011_tokenized.txt','w')
f_tokenizer_in_report = open('microblog2011_tokenized_report.txt','w')

num = 0

for text_line in all_text:

    tokenization_results = word_tokenize(text_line)

    f_tokenizer.write(str(tokenization_results)+'\n')

    if(num<20):
        f_tokenizer_in_report.write(str(num)+":\n")
        f_tokenizer_in_report.write(text_line)
        f_tokenizer_in_report.write(str(tokenization_results)+'\n')
    
    num += 1

f_tokenizer.close()
f_tokenizer_in_report.close()


# capitalize the word
title = "it's"
print(title)
title = title.capitalize()
print(title)
title = title.lower()
print(title)

# handling the special case demo
print("What's the problem, He is doing his assgnments and It's tedious!!")
doc = nlp(u"What's the problem, He is doing his assgnments and It's tedious!!")
print([w.text for w in doc])

# special case is sensitive
with open("transfered_special_case_2.txt","r") as f:
    for line in f:
        special_case = []
        splitted_line = line.split(" ")
        for item_num in range(1,len(splitted_line)-1):
            special_case.append({ORTH:splitted_line[item_num]})
        nlp.tokenizer.add_special_case(splitted_line[0], special_case)

# special_case = [{ORTH:u"He", LEMMA:u"He", POS:u"PRONOUN"}, {ORTH:u"is"}]
# nlp.tokenizer.add_special_case(u"He's", special_case)
# special_case = [{ORTH:u"What", LEMMA:u"What", POS:u"PRONOUN"}, {ORTH:u"is"}]
# nlp.tokenizer.add_special_case(u"What's", special_case)
# special_case = [{ORTH:u"It", LEMMA:u"It", POS:u"PRONOUN"}, {ORTH:u"is"}]
# nlp.tokenizer.add_special_case(u"It's", special_case)

tokens_example = [w.text.lower() for w in nlp(u"What's the problem, He is doing his assgnments and It's tedious!!")]


print(tokens_example)

from gensim.corpora import Dictionary

dct = Dictionary([tokens_example])  # initialize a Dictionary
print(dct)
unique_token_numbers = len(dct)
print("unique_token_numbers:",unique_token_numbers)
dct.save_as_text('foobar.txt')
bag_of_words = dct.doc2bow(tokens_example)
print(bag_of_words)
token_numbers = 0
for i in bag_of_words:
    token_numbers += i[1]
print("token_numbers:",token_numbers)

ratio = unique_token_numbers / token_numbers

print("type/token ratio:", ratio)

# write the result into .txt file
# output_result_file = open("microblog2011_tokenized.txt","w")
# output_result_file.write("sth to write")
# output_result_file.close()

stopwords_list = []
with open('stopwords.txt','r') as f:
    for line in f:
        stopwords_list.append(line[:-1])
# delete /n line break

punctuations="!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

punctuation_list = []
for punc in punctuations:
    punctuation_list.append(punc)

stopwords_with_punctuations = stopwords_list + punctuation_list

filtered_sentence = []
for w in tokens_example: 
    if w not in stopwords_with_punctuations: 
        filtered_sentence.append(w)
print(filtered_sentence)



# create special case text file

f_special_case = open("transfered_special_case.txt","w")

with open("special_case.txt","r") as f:
    for line in f:
        transfered_special_case_example = line.split("/")[0].replace('"','').replace(':','').replace(',','')
        f_special_case.write(transfered_special_case_example+"\n") 
        #just use the most common case

f_special_case.close()

f_special_case = open("transfered_special_case.txt","r")

with open('transfered_special_case_2.txt', 'w') as f_special_case_2:
    for line in f_special_case:
        if not line.strip():
            continue  # skip the empty line
        line_split = line.split(" ")
        capitalized_line = ""
        for l in range(len(line_split)):
            if l == 0 or l == 1:
                capitalized_line += line_split[l].capitalize()
            else:
                capitalized_line += line_split[l]
            if l < len(line_split)-1:
                capitalized_line += " "
        f_special_case_2.write(line)  # non-empty line. Write it to output
        f_special_case_2.write(capitalized_line)

f_special_case.close()