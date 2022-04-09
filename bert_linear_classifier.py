import sst
from datasets import load_dataset
import datasets
import os
import pandas as pd
from tqdm import tqdm, trange
import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

checkpoint = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

dataset = load_dataset("SetFit/sst5", "default")
max_length = 128

# coercing huggingface dataset to torch format
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# creating train, test, validation
train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]
train_data_loader = torch.utils.data.DataLoader(train, batch_size=32)

print(next(iter(train_data_loader)))