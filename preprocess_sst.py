#import sst.load_sst
import pickle as pk
import re
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from gensim.scripts.glove2word2vec import glove2word2vec

lemmatizer = WordNetLemmatizer()
def cleaning(train_x_list, stop_words):
    #train_x_list.apply(lambda x:' '.join(x.lower() for x in x.split())) already lowered in load_sst.py
    big_list = list()
    for sentence in train_x_list:
        # Replacing the special characters
        #sentence = re.sub(r'[^\w\s](([dts]|ll|ve|re)\s)*', '', sentence)
        # # Replacing the digits/numbers
        # new_sent = re.sub(r'\d', '', new_sent)
        # Removing stop words
        all_tokens = list()
        for token in sentence.split():
            if token in stop_words:
                continue
            all_tokens.append(lemmatizer.lemmatize(token))
        big_list.append(' '.join(all_tokens))
    return big_list

train_sst = sst.load_sst.get_train()
uncleaned_train_x = list(train_sst.keys())
train_x = cleaning(uncleaned_train_x, stop_words)
train_y = list(train_sst.values())
pk.dump(train_x, open( "./data/train_x.pk", "wb" ))
pk.dump(train_y, open( "./data/train_y.pk", "wb" ))
pk.dump(uncleaned_train_x, open( "./data/train_x_uncleaned.pk", "wb" ))

test_sst = sst.load_sst.get_test()
uncleaned_test_x = list(test_sst.keys())
test_x = cleaning(uncleaned_test_x, stop_words)
test_y = list(test_sst.values())
pk.dump(test_x, open( "./data/test_x.pk", "wb" ))
pk.dump(test_y, open( "./data/test_y.pk", "wb" ))
pk.dump(uncleaned_test_x, open( "./data/test_x_uncleaned.pk", "wb" ))
