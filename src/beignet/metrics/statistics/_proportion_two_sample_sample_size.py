import torch
from torch import Tensor
from torchmetrics import Metric

import beignet


class ProportionTwoSampleSampleSize(Metric):
    """TorchMetrics wrapper for two-sample proportion sample size calculation.

    This metric accumulates proportion pairs across batches, then computes
    the required sample size to achieve specified power for detecting
    a difference between the two groups.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power (probability of correctly detecting the difference).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".
    ratio : float, default=1.0
        Ratio of sample sizes n2/n1. Default is 1.0 for equal sample sizes.

    Examples
    --------
    >>> metric = ProportionTwoSampleSampleSize(power=0.8, alpha=0.05)
    >>> metric.update(p1=torch.tensor([0.5, 0.4]), p2=torch.tensor([0.6, 0.5]))
    >>> metric.compute()
    tensor([387, 620])
    """

    is_differentiable: bool = True
    higher_is_better: bool = False  # Smaller sample sizes are better
    full_state_update: bool = False

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.power = power
        self.alpha = alpha
        self.alternative = alternative
        self.ratio = ratio

        self.add_state("p1_values", default=[], dist_reduce_fx="cat")
        self.add_state("p2_values", default=[], dist_reduce_fx="cat")

    def update(self, p1: Tensor, p2: Tensor) -> None:
        """Update the metric state with new proportion pairs.

        Parameters
        ----------
        p1 : Tensor
            Proportion in group 1.
        p2 : Tensor
            Proportion in group 2.
        """
        self.p1_values.append(p1.detach())
        self.p2_values.append(p2.detach())

    def compute(self) -> Tensor:
        """Compute the required sample sizes for the accumulated data.

        Returns
        -------
        Tensor
            Required sample size values for group 1.
            Sample size for group 2 = ratio * sample_size_group1.
        """
        if not self.p1_values:
            return torch.tensor([], dtype=torch.float32)

        p1_tensor = torch.cat(self.p1_values, dim=0)
        p2_tensor = torch.cat(self.p2_values, dim=0)

        return beignet.proportion_two_sample_sample_size(
            p1=p1_tensor,
            p2=p2_tensor,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
            ratio=self.ratio,
        )
