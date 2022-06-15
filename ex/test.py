import torch

x = torch.Tensor([-4.0]).double()
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
z.backward()
xpt, ypt = x, y

print( xpt.grad.item(),ypt.data.item())
