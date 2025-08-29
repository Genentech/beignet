"""Chi-square independence sample size metric."""

import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._chi_squared_independence_sample_size import (
    chi_square_independence_sample_size,
)


class ChiSquareIndependenceSampleSize(Metric):
    """Computes required sample size for chi-square independence tests.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. For independence tests, this measures the strength
        of association between two categorical variables. It can be calculated as
        w = √(χ²/n) where χ² is the chi-square statistic and n is the sample size.
    rows : Tensor
        Number of rows in the contingency table (categories of first variable).
    cols : Tensor
        Number of columns in the contingency table (categories of second variable).
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> metric = ChiSquareIndependenceSampleSize()
    >>> effect_size = torch.tensor(0.3)
    >>> rows = torch.tensor(3)
    >>> cols = torch.tensor(3)
    >>> metric.update(effect_size, rows, cols)
    >>> metric.compute()
    tensor(124)
    """

    full_state_update: bool = False

    def __init__(self, power: float = 0.8, alpha: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.power = power
        self.alpha = alpha
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("rows", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("cols", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, rows: Tensor, cols: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Cohen's w effect size.
        rows : Tensor
            Number of rows in contingency table.
        cols : Tensor
            Number of columns in contingency table.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.rows = rows
            self.cols = cols
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.rows = (self.rows * self.total + rows) / (self.total + 1)
            self.cols = (self.cols * self.total + cols) / (self.total + 1)
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the required sample size."""
        if self.total == 0:
            raise RuntimeError("No values have been added to the metric.")

        return chi_square_independence_sample_size(
            input=self.effect_size,
            rows=self.rows,
            cols=self.cols,
            power=self.power,
            alpha=self.alpha,
        )
