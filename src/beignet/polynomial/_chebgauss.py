import operator

import torch


def chebgauss(input):
    ideg = operator.index(input)

    if ideg <= 0:
        raise ValueError

    output = torch.arange(1, 2 * ideg, 2) / (2.0 * ideg)

    output = output * torch.pi

    output = torch.cos(output)

    weight = torch.ones(ideg) * (torch.pi / ideg)

    return output, weight
