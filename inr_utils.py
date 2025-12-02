import numpy as np
from numpy._typing._array_like import NDArray
import torch
from skimage.metrics import structural_similarity

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

# converts values from [-1, 1] to [0, 1] so they can be displayed
def convert_pixel_value_range(values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
  return (values + 1) / 2

def ssim(ground_truth: np.ndarray, model_output: np.ndarray, data_range: float = 2.0) -> float:
    ground_truth = np.clip(ground_truth, -1.0, 1.0, dtype=np.float32)
    model_output = np.clip(model_output, -1.0, 1.0, dtype=np.float32)

    return structural_similarity(ground_truth, model_output, data_range=data_range, channel_axis=2)