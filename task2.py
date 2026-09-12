import torch 
from torch.utils.data import DataLoader, random_split, Subset
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

