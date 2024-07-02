from typing import Literal

from torch import Tensor

from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__z_series_mul import _z_series_mul
from .__z_series_to_c_series import _z_series_to_c_series


def chebmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

    a = _c_series_to_z_series(input)
    b = _c_series_to_z_series(other)

    output = _z_series_mul(a, b, mode=mode)

    output = _z_series_to_c_series(output)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output
