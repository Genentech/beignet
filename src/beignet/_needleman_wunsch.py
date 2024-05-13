import math
import operator

import torch
import torch.func
import torch.nn.functional
from torch import Tensor


def needleman_wunsch(
    input: Tensor,
    lengths: Tensor,
    gap_penalty: float = 0.0,
    temperature: float = 1.0,
):
    def fn(x: Tensor, shape: Tensor) -> Tensor:
        padded = torch.nn.functional.pad(x, [1, 0, 1, 0])

        i = torch.add(
            torch.subtract(
                torch.arange(
                    padded.size(1),
                )[None, :],
                torch.flip(
                    torch.arange(
                        padded.size(0),
                    ),
                    dims=[0],
                )[:, None],
            ),
            operator.sub(
                padded.size(0),
                1,
            ),
        )

        j = torch.floor_divide(
            torch.add(
                torch.flip(
                    torch.arange(
                        padded.size(0),
                    ),
                    dims=[0],
                )[:, None],
                torch.arange(
                    padded.size(1),
                )[None, :],
            ),
            2,
        )

        m = operator.sub(
            operator.add(
                padded.size(0),
                padded.size(1),
            ),
            1,
        )

        n = operator.floordiv(
            operator.add(
                padded.size(0),
                padded.size(1),
            ),
            2,
        )

        y = torch.zeros([m, n], dtype=padded.dtype)

        initialization = torch.zeros(
            [
                padded.size(0),
                padded.size(1),
            ],
            dtype=padded.dtype,
        )

        initialization[:, 0] = torch.multiply(
            torch.arange(padded.size(0)),
            gap_penalty,
        )

        initialization[0, :] = torch.multiply(
            torch.arange(padded.size(1)),
            gap_penalty,
        )

        initialization = y.index_put(
            [i, j],
            initialization,
        )

        previous = torch.zeros(n)

        previous_previous = torch.zeros(n)

        traceback = torch.zeros([m, n])

        mask = y.index_put(
            [i, j],
            torch.nn.functional.pad(
                torch.multiply(
                    torch.less(
                        torch.arange(x.size(0)),
                        shape[0],
                    )[:, None],
                    torch.less(
                        torch.arange(x.size(1)),
                        shape[1],
                    )[None, :],
                ),
                [1, 0, 1, 0],
            ).to(x.dtype),
        )

        striped_indexes = torch.fmod(
            torch.add(
                torch.arange(m),
                math.fmod(
                    padded.size(0),
                    2,
                ),
            ),
            2,
        )

        padded = y.index_put([i, j], padded)

        for index in range(m):
            # TRACEBACK:
            traceback[index] = torch.add(
                # APPLY MASK:
                torch.multiply(
                    # SMOOTH:
                    torch.multiply(
                        # APPLY SOFTMAX:
                        torch.logsumexp(
                            torch.divide(
                                torch.stack(
                                    [
                                        # ALIGN:
                                        torch.add(
                                            previous_previous,
                                            padded[index],
                                        ),
                                        previous + gap_penalty,
                                        # CHANGE DIRECTION:
                                        torch.add(
                                            torch.add(
                                                # INSERT:
                                                torch.multiply(
                                                    torch.nn.functional.pad(
                                                        previous[:-1],
                                                        [1, 0],
                                                    ),
                                                    striped_indexes[index],
                                                ),
                                                # DELETE:
                                                torch.multiply(
                                                    torch.nn.functional.pad(
                                                        previous[+1:],
                                                        [0, 1],
                                                    ),
                                                    operator.sub(
                                                        1,
                                                        striped_indexes[index],
                                                    ),
                                                ),
                                            ),
                                            gap_penalty,
                                        ),
                                    ],
                                ),
                                temperature,
                            ),
                            dim=0,
                        ),
                        temperature,
                    ),
                    mask[index],
                ),
                initialization[index],
            )

            previous_previous, previous = previous, traceback[index]

        return traceback[i, j][shape[0], shape[1]]

    output = torch.empty_like(input)

    for index in range(input.shape[0]):
        output[index] = torch.func.grad(fn)(input[index], lengths[index])

    return output
