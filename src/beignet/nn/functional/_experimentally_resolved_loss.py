import torch
from torch import Tensor


def sigmoid_cross_entropy(input, target):
    logits_dtype = input.dtype
    input = input.double()
    target = target.double()
    log_p = torch.nn.functional.logsigmoid(input)
    log_not_p = torch.nn.functional.logsigmoid(-1 * input)
    output = (-1.0 * target) * log_p - (1.0 - target) * log_not_p
    output = output.to(dtype=logits_dtype)
    return output


def experimentally_resolved_loss(
    input: Tensor,
    atom37_atom_exists: Tensor,
    all_atom_mask: Tensor,
    resolution: Tensor,
    minimum_resolution: float,
    maximum_resolution: float,
    eps: float = 1e-8,
) -> Tensor:
    r"""
    The model contains a head that predicts if an atom is experimentally
    resolved in a high-resolution structure. The input for this head is the
    single representation $\{\mathrm{s}_{i}\}$ produced by the Evoformer stack.
    The single representation is projected with a linear layer and a sigmoid to
    atom-wise probabilities $\{p^{\mathrm{experimentally\;resolved,\;a}}_{i}\}$
    with $i\in\left[1,\ldots,N_{\textrm{residue}}\right]$ and
    $a\in\textsl{S}_{\mathrm{amino\;acids}}$.

    $$\mathcal{L}_{\mathrm{experimentally\;resolved}}=\mathrm{mean}_{\left(i,a\right)}\left(-y_{i}^{a}\log{\left(p_{i}^{\mathrm{experimentally\;resolved},a}\right)}-\left(1-y_{i}^{a}\right)\log{\left(1-p_{i}^{\mathrm{experimentally\;resolved},a}\right)}\right)$$

    where $y_{i}^{a}\in\left\{0,1\right\}$ is the target (i.e., if atom $a$ in
    residue $i$ was resolved in the experiment).
    """
    epsilon = torch.finfo(input.dtype).eps
    errors = sigmoid_cross_entropy(input, all_atom_mask)

    output = errors * atom37_atom_exists
    output = torch.sum(output, dim=-1)
    b = torch.sum(atom37_atom_exists, dim=[-1, -2])
    b = b.unsqueeze(-1)
    b = b + epsilon

    output = output / b

    output = torch.sum(output, dim=-1)

    x = resolution >= minimum_resolution
    y = resolution <= maximum_resolution

    a = x & y

    output = output * a

    output = torch.mean(output)

    return output
