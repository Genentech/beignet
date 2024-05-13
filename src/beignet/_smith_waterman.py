# @title `smith_waterman`


import torch
import torch.func
from torch import Tensor


def smith_waterman(
    input: Tensor,
    lengths: (int, int),
    gap_penalty: float = 0.0,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute the Smith-Waterman alignment score for two sequences.

    The Smith-Waterman algorithm is a local sequence alignment method used
    to identify regions of similarity between two sequences.

    Parameters
    ----------
    input : Tensor
        The similarity matrix of the two sequences.

    lengths : Sequence[int, int]
        A sequence containing the lengths of the two sequences being aligned.

    gap_penalty : float, optional
        The penalty for creating a gap in alignment. Default is 0.

    temperature : float, optional
        Scaling factor to control the sharpness of the score distribution.
        Default is 1.0.

    Returns
    -------
    Tensor
        Smith-Waterman alignment score for the given sequences.
    """

    def fn(
        input: Tensor,
        lengths: (int, int),
    ) -> Tensor:
        if input.is_complex() or input.is_floating_point():
            initial_value = torch.finfo(input.dtype).min
        else:
            initial_value = torch.iinfo(input.dtype).min

        # BOOLEAN MASK TO IDENTIFY VALID POSITIONS:

        a = torch.arange(input.shape[0], device=input.device)
        b = torch.arange(input.shape[1], device=input.device)
        mask = torch.multiply(
            torch.less(
                a,
                lengths[0],
            )[:, None],
            torch.less(
                b,
                lengths[1],
            )[None, :],
        )

        # MASK INVALID POSITIONS:
        # INVERT MASK TO IDENTIFY INVALID POSITIONS:
        # VALUE APPLIED TO INVALID POSITIONS:

        masked_similarity_matrices = input + ~mask * initial_value

        # EXCLUDED LAST ROW AND COLUMN FROM MASKED SIMILARITY MATRICES:
        x_1 = masked_similarity_matrices[
            : masked_similarity_matrices.shape[0] - 1,
            : masked_similarity_matrices.shape[1] - 1,
        ]

        # INDICES FOR ROTATING THE MATRICES TO ALIGN DIAGONALS FOR SCORING:
        rotation_i = torch.flip(torch.arange(x_1.shape[0]), dims=[0])[:, None]
        rotation_j = torch.arange(x_1.shape[1])[None, :]

        # INDICES FOR SCORING ALIGNMENT PATHS THROUGH THE MATRIX MATRICES:
        indexes_i = rotation_j - rotation_i + x_1.shape[0] - 1
        indexes_j = (rotation_i + rotation_j) // 2

        # DIMENSIONS OF THE SCORING MATRICES:
        scores_shape_0 = x_1.shape[0] + x_1.shape[1] - 1
        scores_shape_1 = (x_1.shape[0] + x_1.shape[1]) // 2

        initial_value = -1e10

        # PREVIOUS SCORING MATRICES:
        previous_scores = torch.full(
            [scores_shape_1], initial_value, device=input.device
        )

        previous_scores = (previous_scores, previous_scores)

        # LIST TO ACCUMULATE SCORING MATRICES FOR EACH STEP OF THE ALIGNMENT:
        scores = []

        # INDICES TO MANAGE ALTERNATING UPDATES:
        current_scores_i = torch.fmod(
            torch.arange(scores_shape_0) + x_1.shape[0] % 2, 2
        )

        # SCORING MATRICES WITH `initial_value`:
        current_scores_j = torch.full(
            [
                scores_shape_0,
                scores_shape_1,
            ],
            initial_value,
            device=input.device,
        )

        # INITIAL SCORING MATRICES WITH VALUES FROM THE MASKED SIMILARITY MATRICES:
        current_scores_j = current_scores_j.index_put([indexes_i, indexes_j], x_1)

        current_scores_j[indexes_i, indexes_j] = x_1

        initial_value = torch.tensor([initial_value], device=input.device)

        # LOOP THROUGH EACH SCORE:
        for current_scores in zip(current_scores_i, current_scores_j, strict=False):
            # SCORE FOR EXTENDING ALIGNMENT WITHOUT A GAP:
            torch_add = previous_scores[0] + current_scores[1]

            # SCORE FOR INTRODUCING A GAP:
            t = previous_scores[1] + gap_penalty

            # SCORE FOR OPENING OR EXTENDING A GAP:
            torch_concatenate = torch.concatenate(
                [
                    initial_value,
                    previous_scores[1][:-1],
                ],
            )

            concatenate = torch.concatenate(
                [
                    previous_scores[1][1:],
                    initial_value,
                ],
            )

            tensor = (
                current_scores[0] * torch_concatenate
                + (1 - current_scores[0]) * concatenate
                + gap_penalty
            )

            # APPLYING GAP PENALTIES:
            current_scores = torch.stack(
                [
                    torch_add,
                    t,
                    tensor,
                    # ORIGINAL SCORE FOR THE POSITION:
                    current_scores[1],
                ],
                dim=-1,
            )

            current_scores = current_scores / temperature

            current_scores = torch.maximum(current_scores, initial_value)

            # LOG-SUM-EXP FOR NUMERICAL STABILITY:
            current_scores = torch.special.logsumexp(current_scores, dim=-1)

            # SCORING MATRICES:
            current_scores = current_scores * temperature

            # UPDATE THE SCORES FOR THE NEXT ITERATION:
            previous_scores = previous_scores[1], current_scores

            # ACCUMULATE UPDATED SCORES:
            scores = [*scores, current_scores]

        scores = torch.stack(scores)[indexes_i, indexes_j]

        # COMBINE THE FINAL SCORES AND THE ORIGINAL SIMILARITY MATRIX:
        scores = scores + masked_similarity_matrices[1:, 1:]

        # ADJUST FOR TEMPERATURE:
        scores = scores / temperature

        score = torch.exp(scores - torch.max(scores))

        # APPLY MASK TO SCORING MATRICES:
        # SMITH-WATERMAN SCORES:
        score = (
            torch.log(torch.sum(score * mask[1:, 1:])) + torch.max(scores)
        ) * temperature

        return score

    return torch.func.vmap(torch.func.grad(fn), (0, 0))(input, lengths)
