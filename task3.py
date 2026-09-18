import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np

row, dk = 4, 8

Q, K, V = torch.rand(row, dk), torch.rand(row,dk), torch.rand(row,dk)

activation = Q @ K.T
final = F.softmax(activation/np.sqrt(dk), dim=-1) @ V

final2 = F.scaled_dot_product_attention(query=Q, key=K, value=V)

print(final,final2)
