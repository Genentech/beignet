import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.statistics


class ProportionSampleSize(Metric):
    """TorchMetrics wrapper for one-sample proportion sample size calculation.

    This metric accumulates proportion pairs across batches, then computes
    the required sample size to achieve specified power for detecting
    a difference from the null hypothesis.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Examples
    --------
    >>> metric = ProportionSampleSize(power=0.8, alpha=0.05)
    >>> metric.update(p0=torch.tensor([0.5, 0.4]), p1=torch.tensor([0.6, 0.5]))
    >>> metric.compute()
    tensor([199, 310])
    """

    is_differentiable: bool = True
    higher_is_better: bool = False  # Smaller sample sizes are better
    full_state_update: bool = False

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.power = power
        self.alpha = alpha
        self.alternative = alternative

        self.add_state("p0_values", default=[], dist_reduce_fx="cat")
        self.add_state("p1_values", default=[], dist_reduce_fx="cat")

    def update(self, p0: Tensor, p1: Tensor) -> None:
        """Update the metric state with new proportion pairs.

        Parameters
        ----------
        p0 : Tensor
            Null hypothesis proportion(s).
        p1 : Tensor
            Alternative hypothesis proportion(s).
        """
        self.p0_values.append(torch.atleast_1d(p0.detach()))
        self.p1_values.append(torch.atleast_1d(p1.detach()))

    def compute(self) -> Tensor:
        """Compute the required sample sizes for the accumulated data.

        Returns
        -------
        Tensor
            Required sample size values.
        """
        if not self.p0_values:
            raise RuntimeError("No values have been added to the metric.")

        p0_tensor = torch.cat(self.p0_values, dim=0)
        p1_tensor = torch.cat(self.p1_values, dim=0)

        return beignet.statistics.proportion_sample_size(
            p0=p0_tensor,
            p1=p1_tensor,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )
