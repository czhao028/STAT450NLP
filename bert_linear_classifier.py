import sst
from datasets import load_dataset
import datasets
import os
import pandas as pd
from tqdm import tqdm, trange
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.optim import Adam
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, BertModel

dataset = load_dataset("SetFit/sst5", "default")

# coercing huggingface dataset to torch format
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoding = tokenizer.encode_plus(
  "sample comment",
  add_special_tokens=True,
  max_length=512,
  return_token_type_ids=False,
  padding="max_length",
  return_attention_mask=True,
  return_tensors='pt',
)

max_length = 50
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# creating train, test, validation
train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]
# train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=True)
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", return_dict = 'True')

sample_batch = ((torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)))

sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape

