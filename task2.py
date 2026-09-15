import torch 
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
import numpy as np

import random
import string

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

lr = 0.005
epoch_no = 50

min_word_length = 3
max_word_length = 10

letter_idx = {letter:idx for idx,letter in enumerate(string.ascii_lowercase)}

def generate_point():
    length = random.randint(min_word_length,max_word_length)

    start_end = random.choice([0,1])

    if start_end:
        mid = ''.join(random.choices(string.ascii_lowercase, k=length - 2))
        c = random.choice(string.ascii_lowercase)
        s = c + mid + c
    else:
        start = ''.join(random.choices(string.ascii_lowercase, k=length - 1))
        c = random.choice([ch for ch in string.ascii_lowercase if ch != start[0]])
        s = start + c

    return s, start_end

def word_to_onehot(word):
    word_to_idx = []
    for char in word:
        word_to_idx.append(letter_idx[char])
    encoded_word = F.one_hot(torch.tensor(word_to_idx), num_classes=26).float()

    pad_length = max_word_length - len(word)
    padding = torch.zeros(pad_length, 26)
    en_word = torch.cat((encoded_word, padding))
    return en_word

dataIn = [generate_point() for _ in range(2000)]

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.labels = [l for _,l in data]
        self.words = [w for w,_ in data]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        if self.transform:
            word = self.transform(word)
        if self.target_transform:
            label = self.target_transform(label)
        return word, label

words = CustomDataset(data=dataIn, transform=word_to_onehot,target_transform=torch.tensor)

ratio = sum(l for _, l in dataIn) / len(dataIn)
print(ratio)

training_words, test_words = random_split(words, [.8,.2])
train_words, valid_words = random_split(training_words, [.8,.2])

train_DataLoader = DataLoader(train_words, batch_size=64, shuffle=True)
valid_DataLoader = DataLoader(valid_words, batch_size=64)
test_DataLoader = DataLoader(test_words, batch_size=64)


