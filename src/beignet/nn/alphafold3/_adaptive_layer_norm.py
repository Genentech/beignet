import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module


class AdaptiveLayerNorm(Module):
    r"""
    Adaptive LayerNorm from AlphaFold 3 Algorithm 26.

    This implements the AdaLN operation that conditions layer normalization
    on an additional conditioning signal 's'. The conditioning signal is used
    to adaptively modulate the normalized activations.

    Parameters
    ----------
    c : int
        Channel dimension for the input tensor 'a'
    c_s : int
        Channel dimension for the conditioning signal 's'

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AdaptiveLayerNorm
    >>> batch_size, seq_len, c, c_s = 2, 10, 256, 384
    >>> module = AdaptiveLayerNorm(c=c, c_s=c_s)
    >>> a = torch.randn(batch_size, seq_len, c)
    >>> s = torch.randn(batch_size, seq_len, c_s)
    >>> a_out = module(a, s)
    >>> a_out.shape
    torch.Size([2, 10, 256])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 26: Adaptive LayerNorm
    """

    def __init__(self, c: int, c_s: int):
        super().__init__()

        self.c = c
        self.c_s = c_s

        # Step 1: LayerNorm(a, scale=False, offset=False)
        self.layer_norm_a = LayerNorm(c, elementwise_affine=False)

        # Step 2: LayerNorm(s, offset=False)
        self.layer_norm_s = LayerNorm(c_s, elementwise_affine=False)

        # Step 3: Linear layers for conditioning
        self.linear_s_sigmoid = Linear(c_s, c, bias=False)
        self.linear_s_scale = Linear(c_s, c, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        r"""
        Forward pass of Adaptive LayerNorm.

        Parameters
        ----------
        a : Tensor, shape=(..., c)
            Input tensor to be normalized
        s : Tensor, shape=(..., c_s)
            Conditioning signal tensor

        Returns
        -------
        a_out : Tensor, shape=(..., c)
            Output tensor after adaptive layer normalization
        """
        return torch.sigmoid(
            self.linear_s_sigmoid(
                self.layer_norm_s(s),
            )
        ) * self.layer_norm_a(a) + self.linear_s_scale(
            self.layer_norm_s(s),
        )
