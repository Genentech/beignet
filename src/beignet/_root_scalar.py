import dataclasses
from typing import Callable, Literal

from torch import Tensor

import beignet
import beignet.func


@dataclasses.dataclass
class RootSolutionInfo:
    converged: Tensor
    iterations: Tensor


def root_scalar(
    func: Callable,
    *args,
    method: Literal["bisect"] | Literal["chandrupatla"] = "chandrupatla",
    implicit_diff: bool = True,
    options: dict,
):
    if method == "bisect":
        solver = beignet.bisect
    elif method == "chandrupatla":
        solver = beignet.chandrupatla
    else:
        raise ValueError(f"method {method} not recognized")

    if implicit_diff:
        solver = beignet.func.custom_scalar_root(solver)

    return solver(func, *args, **options)
