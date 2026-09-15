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

learning_rate = 0.01
epoch_no = 50

min_word_length = 3
max_word_length = 8

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

dataIn = [generate_point() for _ in range(20000)]

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
        length = len(word)
        if self.transform:
            word = self.transform(word)
        if self.target_transform:
            label = self.target_transform(label)
        return word, label, length

words = CustomDataset(data=dataIn, transform=word_to_onehot,target_transform=torch.tensor)

ratio = sum(l for _, l in dataIn) / len(dataIn)
print(ratio)

training_words, test_words = random_split(words, [.8,.2])
train_words, valid_words = random_split(training_words, [.8,.2])

train_DataLoader = DataLoader(train_words, batch_size=64, shuffle=True)
valid_DataLoader = DataLoader(valid_words, batch_size=64)
test_DataLoader = DataLoader(test_words, batch_size=64)


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, word_tensor, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            word_tensor, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        logits = self.h2o(hidden[0])
        return logits

rnn = CharRNN(input_size=26,hidden_size=32,output_size=2)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

test_size = len(test_DataLoader.dataset)
test_num_batches = len(test_DataLoader)

train_loss_list, valid_loss_list = [], []

for epoch in range(epoch_no):

    train_loss, valid_loss, train_correct = 0, 0, 0
    rnn.train()
    for words, labels, lengths in train_DataLoader:
        pred = rnn(words, lengths)
        loss = loss_fn(pred, labels)

        with torch.no_grad():
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        rnn.eval()
        for words, labels, lengths in valid_DataLoader:
            val = rnn(words, lengths)
            valid_loss += loss_fn(val, labels).item()

        train_loss_list.append(train_loss/len(train_DataLoader))
        valid_loss_list.append(valid_loss/len(valid_DataLoader))
        accuracy = train_correct / len(train_DataLoader.dataset)
        print(accuracy)
        if epoch % 5 == 0:
            print(f'EPOCH: {epoch}')
    
# TEST

rnn.eval()
test_loss, correct = 0, 0
for words, labels, lengths in test_DataLoader:
    with torch.no_grad():
        test = rnn(words, lengths)
        test_loss += loss_fn(test, labels).item()
        correct += (test.argmax(1) == labels).type(torch.float).sum().item()

accuracy = correct / test_size
avg_test_loss = test_loss / test_num_batches
print(f'TEST\nAccuracy: {accuracy}\nAverage test loss: {avg_test_loss}\n')

e = np.arange(epoch_no)

plt.figure()
plt.plot(e, train_loss_list)
plt.plot(e, valid_loss_list)
plt.legend(['Train Loss','Valid Loss'])
plt.title('Average train loss and average valid loss against epoch number')
plt.show()