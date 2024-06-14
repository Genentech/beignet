import torch


def safe_mask(mask, fn, operand, placeholder=0):
    masked = torch.where(mask, operand, torch.tensor(0, dtype=operand.dtype))

    return torch.where(mask, fn(masked), torch.tensor(placeholder, dtype=operand.dtype))