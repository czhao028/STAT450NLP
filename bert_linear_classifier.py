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
df_train.rename(columns = {0: 'Sentence'}, inplace = True)
df_test.rename(columns = {0: 'Sentence'}, inplace = True)

def data_to_examples(training, testing, data_column, label_column):
    train_InputExamples = training.apply(lambda x: InputExample(guid=None, text_a = x[data_column],
                                                          text_b = None, label = x[label_column]), axis = 1)

    validation_InputExamples = testing.apply(lambda x: InputExample(guid=None, text_a = x[data_column],
                                                          text_b = None, label = x[label_column]), axis = 1)
    return train_InputExamples, validation_InputExamples

train_InputExamples, validationInputExamples = data_to_examples(df_train, df_test, 0, 1)

def examples_to_tf(examples, tokenizer, max_length = 128):
    features = []
    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

train_InputExamples, validationInputExamples = data_to_examples(df_train, df_test, 0, 1)

train_data = examples_to_tf(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)