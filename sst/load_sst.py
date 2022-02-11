# Load data
import pytreebank
import sys
import os

sst_dataset = pytreebank.load_sst('./raw_data')

def get_label_sent_for_category(category="train"):
    return_list = list()
    for item in sst_dataset[category]:
        return_list.append(item.to_labeled_lines()[0]) #labels sentence & sub-phrases. [0] gets entire sentence
    return return_list

def get_train():
    return get_label_sent_for_category("train")
def get_test():
    return get_label_sent_for_category("test")
def get_dev():
    return get_label_sent_for_category("dev")

