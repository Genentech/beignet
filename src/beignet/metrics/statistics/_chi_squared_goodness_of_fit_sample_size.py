"""Chi-square goodness-of-fit sample size metric."""

import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._chi_squared_goodness_of_fit_sample_size import chi_square_goodness_of_fit_power_sample_size


class ChiSquareGoodnessOfFitSampleSize(Metric):
    """Computes required sample size for chi-square goodness-of-fit tests.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. This is calculated as the square root of the
        sum of squared standardized differences: w = √(Σ((p₁ᵢ - p₀ᵢ)²/p₀ᵢ))
        where p₀ᵢ are the expected proportions and p₁ᵢ are the observed proportions.
    df : Tensor
        Degrees of freedom for the chi-square test (number of categories - 1).
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> metric = ChiSquareGoodnessOfFitSampleSize()
    >>> effect_size = torch.tensor(0.3)
    >>> df = torch.tensor(3)
    >>> metric.update(effect_size, df)
    >>> metric.compute()
    tensor(108)
    """

    full_state_update: bool = False

    def __init__(self, power: float = 0.8, alpha: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.power = power
        self.alpha = alpha
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("df", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, df: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Cohen's w effect size.
        df : Tensor
            Degrees of freedom.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.df = df
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.df = (self.df * self.total + df) / (self.total + 1)
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the required sample size."""
        return chi_square_goodness_of_fit_power_sample_size(
            effect_size=self.effect_size,
            df=self.df,
            power=self.power,
            alpha=self.alpha,
        )
