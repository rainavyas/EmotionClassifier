'''
Prepare the Twitter Emotions dataset as tokenized torch
ids tensor and attention mask
'''

import torch
import torch.nn as nn
import scandir
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer

_DESCRIPTION = """\
Twitter Emotions Dataset
This dataset consists of train (16000), val (2000) and test (2000), where each 
data point is a short tweet. Each tweet is classed into one of 
eight emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust\
"""

_DOWNLOAD_URL = "https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt"


def read_file(filepath, CLASS_TO_IND):
    tweets = []
    class_labels = []

    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    for line in lines:
        items = line.split(';')
        try:
            label = CLASS_TO_IND[items[1]]
            class_labels.append(label)
            tweets.append(items[0])
        except:
            raise print("Failed to convert class", items[1])
    return tweets, class_labels


def get_data(filepath, arch):
    allowed_arch = ['electra', 'bert', 'roberta']
    if arch not in allowed_arch:
        raise Exception('Invalid architecture, only allowed: electra, bert, roberta')
    
    CLASS_TO_IND = {
        'anger': 0,
        'anticipation': 1,
        'disgust': 2,
        'fear': 3,
        'joy': 4,
        'sadness': 5,
        'surprise': 6,
        'trust': 7
    }

    tweets_list, labels = read_file(filepath, CLASS_TO_IND)

    # Tokenize and prep input tensors
    if arch == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    elif arch == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif arch == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    encoded_inputs = tokenizer(tweets_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    labels = torch.LongTensor(labels)

    return ids, mask, labels


def get_train(arch, filepath='../data/train.txt'):
    return get_data(filepath, arch)

def get_val(arch, filepath='../data/val.txt'):
    return get_data(filepath, arch)

def get_test(arch, filepath='../data/test.txt'):
    return get_data(filepath, arch)
