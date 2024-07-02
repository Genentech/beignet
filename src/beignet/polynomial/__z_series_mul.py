from typing import Literal

import torchaudio.functional
from torch import Tensor


def _z_series_mul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    return torchaudio.functional.convolve(input, other, mode=mode)
