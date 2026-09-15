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

def generate_point():
    length = random.randint(3,10)

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

dataIn = [generate_point() for _ in range(2000)]

class CustomImageDataset(Dataset):
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

words = CustomImageDataset(data=dataIn, target_transform=torch.tensor)
