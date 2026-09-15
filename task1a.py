import torch 
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
import numpy as np

# Checking for accelerators and uses if available, else defaults to cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        # Necessary for subclass NN, declares Module init to self
        super().__init__()
        # flatten converts n-dim array into (batch no, features) for parsing
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

full_data = datasets.FashionMNIST(
    root="data",
    # Identifies that training data is retrieved
    train=True,
    # Downloads from internet if not found on device
    download=True,
    # Transforms data to Pytorch image tensor, with pixel intensity values cast to float32
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    # # Converts integer labels to one-hot encoding for compatibility with NNs
    # target_transform=v2.Lambda(
    #     lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    # ),
    # Transforms not necessary when using nn.CrossEntropyLoss()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

training_data, valid_data = random_split(full_data, [.8,.2])

learning_rate = 0.01
epoch_number = 200

train_dataLoader = DataLoader(Subset(training_data, range(500)), batch_size=64, shuffle=True)
valid_dataLoader = DataLoader(valid_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

model = NeuralNetwork().to(device=device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

test_size = len(test_dataLoader.dataset)
test_num_batches = len(test_dataLoader)

train_loss_list, valid_loss_list = [], []

for epoch in range(epoch_number):

    train_loss, valid_loss, train_correct = 0, 0, 0
    model.train()
    for images, labels in train_dataLoader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)

        # Computes cost fn (using CrossEntropyLoss on labels and predictions)
        loss = loss_fn(pred, labels) 

        with torch.no_grad():
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        # Backpropagation, loss.backward() gets gradient
        loss.backward()
        #Optimizer takes step using gradient, SGD
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        model.eval()
        for images, labels in valid_dataLoader:
            images, labels = images.to(device), labels.to(device)
            val = model(images)
            valid_loss += loss_fn(val, labels).item()

        train_loss_list.append(train_loss/len(train_dataLoader))
        valid_loss_list.append(valid_loss/len(valid_dataLoader))
        accuracy = train_correct / len(train_dataLoader.dataset)
        print(accuracy)
        if epoch % 5 == 0:
            print(f'EPOCH: {epoch}')

# TEST

model.eval()
test_loss, correct = 0, 0
for images, labels in test_dataLoader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        test = model(images)
        test_loss += loss_fn(test, labels).item()
        correct += (test.argmax(1) == labels).type(torch.float).sum().item()

accuracy = correct / test_size
avg_test_loss = test_loss / test_num_batches
print(f'TEST\nAccuracy: {accuracy}\nAverage test loss: {avg_test_loss}\n')

e = np.arange(epoch_number)

plt.figure()
plt.plot(e, train_loss_list)
plt.plot(e, valid_loss_list)
plt.legend(['Train Loss','Valid Loss'])
plt.title('Average train loss and average valid loss against epoch number')
plt.show()


# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }


