from dataclasses import dataclass
from typing import Callable, Literal

from torch import Tensor

import beignet
import beignet.func


@dataclass
class RootSolutionInfo:
    converged: Tensor
    iterations: Tensor


def root_scalar(
    f: Callable,
    *args,
    method: Literal["bisect"] = "bisect",
    implicit_diff: bool = True,
    options: dict,
):
    if method == "bisect":
        solver = beignet.bisect
    else:
        raise ValueError(f"method {method} not recognized")

    if implicit_diff:
        solver = beignet.func.implicit_diff_root_scalar(solver)

    return solver(f, *args, **options)
