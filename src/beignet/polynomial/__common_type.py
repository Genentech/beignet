import torch


def _common_type(*xs):
    dtypes = [
        [
            torch.float16,
            torch.float32,
            torch.float64,
        ],
        [
            None,
            torch.complex64,
            torch.complex128,
        ],
    ]

    precisions = {
        torch.float16: 0,
        torch.float32: 1,
        torch.float64: 2,
        torch.complex64: 1,
        torch.complex128: 2,
    }

    is_complex = False

    precision = 0

    for x in xs:
        if torch.is_complex(x):
            is_complex = True

        if torch.is_floating_point(x) or torch.is_complex(x):
            score = precisions.get(x.dtype, None)

            if score is None:
                raise TypeError
        elif torch.is_tensor(x) and x.dtype in precisions:
            score = precisions[x.dtype]
        else:
            score = precisions[torch.float64]

        precision = max(precision, score)

    if is_complex:
        return dtypes[1][precision]
    else:
        return dtypes[0][precision]
