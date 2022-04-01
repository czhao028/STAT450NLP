import pickle as pk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, SpatialDropout1D, LSTM, Activation, Dropout, Dense, Input
from sklearn.model_selection import KFold
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y_1 = np.array(pk.load(open( "./data/train_y.pk", "rb" )))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y_1 = np.array(pk.load(open( "./data/test_y.pk", "rb" )))

import time
t1 = time.time()

maxLen = 300
maxVocabSize = 50000
embedding_dim = 100
tokenizer = Tokenizer(num_words=maxVocabSize)
tokenizer.fit_on_texts(train_x)
words_to_index = tokenizer.word_index
print('Found %s unique tokens.' % len(words_to_index))


train_x_np = tokenizer.texts_to_sequences(train_x)
train_x_np = pad_sequences(train_x_np, maxlen=maxLen)
print('Shape of training data tensor:', train_x_np.shape)
test_x_np = tokenizer.texts_to_sequences(test_x)
test_x_np = pad_sequences(test_x_np, maxlen=maxLen)
print('Shape of testing data tensor:', test_x_np.shape)

#turn y into dummy variable
train_y_np = np.zeros((train_y_1.size, train_y_1.max()+1))
train_y_np[np.arange(train_y_1.size),train_y_1] = 1
test_y_np = np.zeros((test_y_1.size, test_y_1.max()+1))
test_y_np[np.arange(test_y_1.size),test_y_1] = 1

#cross validation for number of neurons
list_neurons = [2,5,15,30,45]

# Define the K-fold Cross Validator
num_folds = 5
num_classes = 5
epochs = 5
batch_size = 32
num_neurons = 15 #chosen by cross-validation
# kfold = KFold(n_splits=num_folds, shuffle=True)
#
# # K-fold Cross Validation model evaluation
# fold_no = 1
#
# for num_neurons in list_neurons:
#     acc_per_fold = list()
#     loss_per_fold = list()
#     for train, test in kfold.split(train_x_np, train_y_np):
#
#       # Define the model architecture
#       model = Sequential()
#       model.add(Embedding(maxVocabSize, embedding_dim, input_length=train_x_np.shape[1]))
#       model.add(SpatialDropout1D(0.2))
#       model.add(LSTM(num_neurons, dropout=0.2, recurrent_dropout=0.2))
#       model.add(Dense(num_classes, activation='softmax'))  # if sigmoid, will only give you 0 --> 1, rather than multiclass
#       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#       # Generate a print
#       print('------------------------------------------------------------------------')
#       print(f'Training for fold {fold_no} ...')
#
#       # Fit data to model
#       model.fit(train_x_np[train], train_y_np[train], epochs=epochs, batch_size=batch_size)
#
#       # Generate generalization metrics
#       scores = model.evaluate(train_x_np[test], train_y_np[test], verbose=0)
#       print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
#       acc_per_fold.append(scores[1] * 100)
#       loss_per_fold.append(scores[0])
#
#       # Increase fold number
#       fold_no = fold_no + 1
#     print("Num neurons: ", num_neurons, "CV accuracy:", sum(acc_per_fold)/len(acc_per_fold))
#     print(acc_per_fold,"/", loss_per_fold)



model = Sequential()
model.add(Embedding(maxVocabSize, embedding_dim, input_length=train_x_np.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(num_neurons, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax')) #if sigmoid, will only give you 0 --> 1, rather than multiclass
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_x_np, train_y_np, epochs = epochs, batch_size=batch_size, validation_split=0.1)

# score, acc = model.evaluate(test_x_np, test_y_np)
#score,acc = model.evaluate(test_x, test_y, verbose = 2, batch_size = batch_size)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))
pred_y_decimal = model.predict(test_x_np) #ROUND pred_y - highest value in each row is 1, all others are 0
pred_y = np.zeros_like(pred_y_decimal)
pred_y[np.arange(len(pred_y_decimal)), pred_y_decimal.argmax(1)] = 1

print(classification_report(test_y_np, pred_y))

pred_y_list = pred_y.argmax(1)

disp = ConfusionMatrixDisplay.from_predictions(test_y_1, pred_y_list,
    display_labels=list(range(1, 6)),
    cmap=plt.cm.Blues,
    normalize="true")

disp.ax_.set_title("Normalized Confusion Matrix: LSTM")
print(disp.confusion_matrix)
plt.savefig("lstm_confusion.png")
plt.show()


print("Total time executing", time.time() - t1)