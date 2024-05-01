import torch


def global_distance_test(input, other, mask, cutoffs):
    n = torch.sum(mask, dim=-1)

    input = input.float()
    other = other.float()

    distances = torch.sqrt(torch.sum((input - other) ** 2, dim=-1))

    scores = []

    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)
