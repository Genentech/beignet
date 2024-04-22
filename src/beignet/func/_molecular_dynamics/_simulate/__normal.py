import torch
from torch import Tensor

from ..__dataclass import _dataclass


@_dataclass
class _Normal:
    mean: Tensor
    var: Tensor

    def sample(self):
        mu, sigma = self.mean, torch.sqrt(self.var)

        return mu + sigma * torch.normal(0.0, 1.0, mu.shape, dtype=mu.dtype)

    def log_prob(self, x):
        return (
            -0.5 * torch.log(2 * torch.pi * self.var)
            - 1 / (2 * self.var) * (x - self.mean) ** 2
        )
