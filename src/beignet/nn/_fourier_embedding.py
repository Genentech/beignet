import torch
import torch.nn as nn
from torch import Tensor


class FourierEmbedding(nn.Module):
    r"""
    Fourier Embedding from AlphaFold 3 Algorithm 22.

    This implements Fourier positional embeddings using random weights and biases
    that are generated once before training and then frozen. The embedding uses
    cosine activation: cos(2π(tw + b))

    Parameters
    ----------
    c : int
        Output embedding dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import FourierEmbedding
    >>> batch_size, seq_len, c = 2, 10, 256
    >>> module = FourierEmbedding(c=c)
    >>> t = torch.randn(batch_size, seq_len, 1)  # Time or position values
    >>> embeddings = module(t)
    >>> embeddings.shape
    torch.Size([2, 10, 256])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 22: Fourier Embedding
    """

    def __init__(self, c: int):
        super().__init__()

        self.c = c

        # Algorithm 22 Step 1: w, b ~ N(0, I_c)
        # Randomly generate weight/bias once before training
        # These are frozen parameters (not updated during training)
        self.register_buffer("w", torch.randn(c))
        self.register_buffer("b", torch.randn(c))

    def forward(self, input: Tensor) -> Tensor:
        r"""
        Forward pass of Fourier Embedding.

        Parameters
        ----------
        input : Tensor, shape=(..., 1) or (...,)
            Input tensor containing times and positions.

        Returns
        -------
        embeddings : Tensor, shape=(..., c)
            Fourier embeddings using cosine activation
        """
        # Handle both (..., 1) and (...,) input shapes
        if input.dim() > 1 and input.shape[-1] == 1:
            input = input.squeeze(-1)  # Remove last dimension if it's 1

        # Ensure t has correct shape for broadcasting
        # t should be (...,) and we want to broadcast with w and b which are (c,)
        input = torch.unsqueeze(input, -1)  # Shape: (..., 1)

        # Algorithm 22 Step 2: return cos(2π(tw + b))
        # Broadcasting: (..., 1) * (c,) + (c,) -> (..., c)
        return torch.cos(2 * torch.pi * (input * self.w + self.b))
