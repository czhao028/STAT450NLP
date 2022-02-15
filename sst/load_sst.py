# Load data
import pytreebank
import sys
import os


raw_data_path = os.path.join(sys.path[0], 'sst\\raw_data')
sst_dataset = pytreebank.load_sst(raw_data_path)

def get_label_sent_for_category(category="train"):
    return_dict = {}
    for item in sst_dataset[category]:
        label, sent = item.to_labeled_lines()[0]
        return_dict[sent.lower()] = label #labels sentence & sub-phrases. [0] gets entire sentence
    return return_dict

def get_train():
    return get_label_sent_for_category("train")
def get_test():
    return get_label_sent_for_category("test")
def get_dev():
    return get_label_sent_for_category("dev")



#get_train()
