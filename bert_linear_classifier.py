import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification, InputExample, InputFeatures
import os
import pickle as pk

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# creating training and testing sets as pandas dataframes
train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y = pk.load(open( "./data/train_y.pk", "rb" ))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y = pk.load(open( "./data/test_y.pk", "rb" ))

df_test_y = pd.DataFrame(test_y) + 1
df_train_y = pd.DataFrame(train_y) + 1
df_test_x = pd.DataFrame(test_x)
df_train_x = pd.DataFrame(train_x)

df_test_y.rename(columns={0: 'Rating'}, inplace=True)
df_train_y.rename(columns={0: 'Rating'}, inplace=True)

df_test = pd.concat([df_test_y.reset_index(drop=True), df_test_x], axis=1)
df_test.rename(columns = {0: 'Sentence'}, inplace = True)

df_train = pd.concat([df_train_y.reset_index(drop=True), df_train_x], axis=1)
df_test.rename(columns = {0: 'Sentence'}, inplace = True)

#def data_to_examples(training, testing, data_column, label_column):
#    train_InputExamples = training.apply(lambda x: InputExample(guid=None, text_a = x[data_column],
#                                                          text_b = None, label = x[label_column]), axis = 1)

    #validation_InputExamples = testing.apply(lambda x: InputExample(guid=None, text_a = x[data_column],
    #                                                      text_b = None, label = x[label_column]), axis = 1)
#    return train_InputExamples, y

#train_InputExamples = data_to_examples(df_train, df_test, 0, 1)

#train_InputExamples = df_train.apply(lambda x: InputExample(guid=None, text_a = x[0],
#                                                          text_b = None, label = x[1]), axis = 1)

#print(train_InputExamples.head())
print(df_train[0].head())