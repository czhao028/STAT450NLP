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

dataset = load_dataset("sst", "default", encoding='latin-1')
train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]

