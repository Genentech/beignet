import torch


def _nonzero(
    input,
    size=1,
    fill_value=0,
):
    output = torch.nonzero(input, as_tuple=False)

    if output.shape[0] > size:
        output = output[:size]
    elif output.shape[0] < size:
        output = torch.concatenate(
            [
                output,
                torch.full(
                    [
                        size - output.shape[0],
                        output.shape[1],
                    ],
                    fill_value,
                ),
            ],
            0,
        )

    return output
