import sst.load_sst
import pickle as pk
import re
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
def cleaning(train_x_list, stop_words):
    #train_x_list.apply(lambda x:' '.join(x.lower() for x in x.split())) already lowered in load_sst.py
    for sentence in train_x_list:
        # Replacing the special characters
        new_sent = re.sub(r'[^\w\s](([dts]|ll|ve|re)\s)*', '', sentence)
        # # Replacing the digits/numbers
        # new_sent = re.sub(r'\d', '', new_sent)
        # Removing stop words
        all_tokens = list()
        for token in new_sent.split():
            if token in stop_words:
                continue
            all_tokens.append(lemmatizer.lemmatize(token))
        return ' '.join(all_tokens)

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
