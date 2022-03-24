import pickle as pk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, SpatialDropout1D, LSTM, Activation, Dropout, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y = np.array(pk.load(open( "./data/train_y.pk", "rb" )))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y = np.array(pk.load(open( "./data/test_y.pk", "rb" )))

import time
t1 = time.time()

def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      try:
        word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
      except Exception as e:
        print(curr_word, e)
  return word_to_vec_map

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
words_to_index = tokenizer.word_index

word_to_vec_map = read_glove_vector('glove.6B.50d.txt')
maxLen = 300

vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]
emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
  embedding_vector = word_to_vec_map.get(word)
  if embedding_vector is not None:
    emb_matrix[index-1, :] = embedding_vector

embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)

lstm_out = 196
model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax')) #if sigmoid, will only give you 0 --> 1, rather than multiclass
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

#convert training & testing sentences into GloVe word embeddings
X_train_indices = tokenizer.texts_to_sequences(train_x)
X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
X_test_indices = tokenizer.texts_to_sequences(test_x)
X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

batch_size = 32
model.fit(X_train_indices, train_y, epochs = 7, batch_size=batch_size, verbose = 2)
score, acc = model.evaluate(X_test_indices, test_y, verbose=2, batch_size=batch_size)
#score,acc = model.evaluate(test_x, test_y, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))