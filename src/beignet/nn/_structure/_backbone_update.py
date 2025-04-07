from typing import Tuple

from torch import Tensor
from torch.nn import Linear, Module


class BackboneUpdate(Module):
    """
    The updates for the backbone frames are created by predicting a quaternion
    for the rotation and a vector for the translation. The first component of
    the non-unit quaternion is fixed to 1. The three components defining the
    Euler axis are predicted by the network. This procedure guarantees a valid
    normalized quaternion and furthermore favours small rotations over large
    rotations (the quaternion (1, 0, 0, 0) is the identity rotation).
    """

    def __init__(self, c_s):
        super().__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        input: Tensor
            [*, N_res, C_s] single representation

        Returns
        -------
        Tuple[Tensor, Tensor]
            [*, N_res, 6] quaternion and translation vector
        """
        return self.linear(input)
