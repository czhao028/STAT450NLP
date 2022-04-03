import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import bert
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification, InputExample, InputFeatures
import os
import pickle as pk

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
df_train.rename(columns = {0: 'Sentence'}, inplace = True)
df_test.rename(columns = {0: 'Sentence'}, inplace = True)

numeric_feature_names = ["Rating"]
numeric_features = df_train[numeric_feature_names]

target = df_train.pop('Sentence')

training_df = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))

# Preprocessing, modeling, and results
bert_preprocess_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_model = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2')

def build_bert_classifier():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
  tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

classifier_model = build_bert_classifier()
#bert_raw_result = classifier_model(tf.constant(text_test))
#print(tf.sigmoid(bert_raw_result))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = tf.metrics.Accuracy()

epochs = 5
# ---------------------------------------------------------------

steps_per_epoch = tf.data.experimental.cardinality(training_df).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

#history = classifier_model.fit(x=train_ds,
#                               validation_data=val_ds,
#                               epochs=epochs)

#loss, accuracy = classifier_model.evaluate(test_ds)

#print(f'Loss: {loss}')
#print(f'Accuracy: {accuracy}')