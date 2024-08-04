from typing import Callable, Literal

import beignet
import beignet.func


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
