import torch

# define differential operators that allow us to leverage autograd to compute gradients, the laplacian, etc.
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = torch.zeros_like(y[..., :1])
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def map_to_01(A,flag=False):
  '''Change the pixel range of an image from -1 ~ 1 to 0 ~ 1. If flag, 
  the original range depends on min and max values of the pixels of the image
  A: torch.tensor
  flag: boolean'''
  if flag: 
    A -= torch.min(A)
    A /= torch.max(A)
  else:
    A = A + 1
    A = A/2
  return A