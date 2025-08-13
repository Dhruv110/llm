import torch
import torch.nn.functional as F

y = torch.tensor([1.0])  #True Lable
x1 = torch.tensor([1.1])   #Input Feature
w1 = torch.tensor([2.2], requires_grad=True)   #Weight Parameter
b = torch.tensor([0.0], requires_grad=True)   #Bias Unit

z = x1 * w1 + b  #Net Input
a = torch.sigmoid(z)   #activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)
