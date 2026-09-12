import torch 
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(43)

x = torch.arange(-5,5,0.05)

N = len(x)

mu, sigma = 0, 0.1 # mean and standard deviation
noise = torch.randn(len(x)) * sigma + mu

lr = torch.tensor([0.1])

y = 3*x + 2 + noise

print(noise[:3])

# tensor containing parameters, requires_grad=True tracks all 
# operations performed using params
params = torch.randn(2, requires_grad=True)

# similar to best fit line, puts x values and 1s into matrix for linalg methods
x_aug = torch.stack([x,torch.ones_like(x)], dim=1)

print(x_aug[:3])

iters = np.arange(1,21,1)
loss_list = []

for i in iters:
    y_pred = x_aug @ params # Matrix product of x-values and parameters 
    L = ((y_pred - y)**2).mean() # MSE
    loss_list.append(L.item())
    # Instead of analytically determining grad function by hand, .backward() computes 
    # grad using requires_grad
    L.backward()
    # Finding grad of w-b space i.e. partial derivates of L wrt w, b
    # as L depends on w/b through every y_pred,i, need to take sum of all y_pred,i
    if i % 5 == 0:
        print("Loss: " + str(L.item())) 
    # grad = torch.stack([((2/N)*(y_pred - y)*x).sum(), ((2/N)*(y_pred - y)).sum()], dim=0)
    
    # torch.no_grad() stops autograd from tracking this particular operation as it is 
    # not to be included in grad path
    with torch.no_grad():
        params -= lr*params.grad

    # Zeroes grad numbers for next epoch, if not will hold previous value
    # No requirement for torch.no_grad as params.grad is a separate tensor with requires_grad=False
    params.grad.zero_()

print(str(params))

plt.figure()
plt.stem(iters, loss_list)
plt.title('Loss against iteration no')
plt.show()

