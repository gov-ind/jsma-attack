import torch
from pdb import set_trace

set_trace()

x = torch.Tensor([1, 2, 3])
#x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
x.requires_grad = True

def model(x): return torch.tensor([x ** 2, x ** 3])

def model(x): return torch.stack([x ** 2, x ** 3])

a1 = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

def model(x):
    return x ** 2 + torch.flip(x, dims=(0,))

'''def model(x):
  return torch.stack([
    torch.tensor([(x[0] ** 2).item(), x[1].item(), x[2].item()]),
    x ** 3
  ])'''

y = model(x)

#grads = [torch.autograd.grad(outputs=out, inputs=x, retain_graph=True)[0][i] 
#for i, out in enumerate(y)]

jac = []

'''for i in range(y.shape[0]):
  a1 = []
  for j in range(y.shape[1]):
    a1.append(torch.autograd.grad(outputs=y[i][j], inputs=x, retain_graph=True)[0][j])
  jac.append(a1)
'''
set_trace()

jac2 = torch.autograd.functional.jacobian(model, x)
