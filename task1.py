import torch 
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn

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
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

training_data = datasets.FashionMNIST(
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

learning_rate = 0.004
epoch_number = 20

train_dataLoader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=64)

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

test_size = len(test_dataLoader.dataset)
test_num_batches = len(test_dataLoader)

for epoch in range(epoch_number):

    test_loss, correct = 0, 0

    for images, labels in train_dataLoader:
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for images, labels in test_dataLoader:
        with torch.no_grad():
            valid = model(images)
            test_loss += loss_fn(valid, labels).item()
            correct += (valid.argmax(1) == labels).type(torch.float).sum().item()

    accuracy = correct / test_size
    avg_test_loss = test_loss / test_num_batches
    print(f'Epoch {epoch + 1}\nAccuracy: {accuracy}\nAverage test loss: {avg_test_loss}\n')



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


