from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
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
# all_text =  open('twitter_corpus.txt', 'r', encoding='utf-8-sig')
# for text_line in all_text:
#     print(text_line)

#     tokenization_results = word_tokenize(text_line)
    
#     print(tokenization_results)



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
special_case = [{ORTH:u"He", LEMMA:u"He", POS:u"PRONOUN"}, {ORTH:u"is"}]
nlp.tokenizer.add_special_case(u"He's", special_case)
special_case = [{ORTH:u"What", LEMMA:u"What", POS:u"PRONOUN"}, {ORTH:u"is"}]
nlp.tokenizer.add_special_case(u"What's", special_case)
special_case = [{ORTH:u"It", LEMMA:u"It", POS:u"PRONOUN"}, {ORTH:u"is"}]
nlp.tokenizer.add_special_case(u"It's", special_case)

print([w.text for w in nlp(u"What's the problem, He is doing his assgnments and It's tedious!!")])

# write the result into .txt file
output_result_file = open("microblog2011_tokenized.txt","w")
output_result_file.write("sth to write")
output_result_file.close()