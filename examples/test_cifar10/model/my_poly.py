import torch
import torch.nn as nn

class MyPolyReLU(nn.Module):
    """Hermite degree-4 polynomial approximation of ReLU."""
    def __init__(self, hermite_coeffs=None, upper_bound=None):
        super().__init__()
        # 忽略传入的系数，使用内置的 Hermite 系数
        self.register_buffer('c0', torch.tensor(0.39894228))
        self.register_buffer('c1', torch.tensor(0.5))
        self.register_buffer('c2', torch.tensor(0.19947114))
        self.register_buffer('c4', torch.tensor(-0.01662260))

    def forward(self, x):
        return self.c0 + self.c1 * x + self.c2 * x * x + self.c4 * x * x * x * x
