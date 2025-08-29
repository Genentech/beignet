import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.statistics


class ProportionTwoSamplePower(Metric):
    """TorchMetrics wrapper for two-sample proportion power calculation.

    This metric accumulates proportions and sample sizes for two groups across batches,
    then computes the statistical power for detecting a difference between the groups.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Examples
    --------
    >>> metric = ProportionTwoSamplePower(alpha=0.05)
    >>> p1 = torch.tensor([0.5, 0.4])
    >>> p2 = torch.tensor([0.6, 0.5])
    >>> n1 = torch.tensor([100, 150])
    >>> n2 = torch.tensor([100, 150])
    >>> metric.update(p1=p1, p2=p2, n1=n1, n2=n2)
    >>> metric.compute()
    tensor([0.5592, 0.6428])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.alternative = alternative

        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: Tensor,
        sample_size: Tensor,
    ) -> None:
        """Update the metric state with new effect size and sample size values.

        Parameters
        ----------
        effect_size : Tensor
            Effect size (difference in proportions).
        sample_size : Tensor
            Sample size per group.
        """
        self.effect_size_values.append(torch.atleast_1d(effect_size))
        self.sample_size_values.append(torch.atleast_1d(sample_size))

    def compute(self) -> Tensor:
        """Compute the statistical power for the accumulated data.

        Returns
        -------
        Tensor
            Statistical power values.
        """
        if not self.effect_size_values or not self.sample_size_values:
            raise RuntimeError("No values have been added to the metric.")

        effect_size_tensor = torch.cat(self.effect_size_values, dim=0)
        sample_size_tensor = torch.cat(self.sample_size_values, dim=0)

        # Convert effect size to p1, p2 assuming baseline p1=0.5 and p2 = p1 + effect_size
        p1 = torch.full_like(effect_size_tensor, 0.5)
        p2 = p1 + effect_size_tensor
        p2 = torch.clamp(p2, 0.0, 1.0)  # Ensure valid probabilities

        return beignet.statistics.proportion_two_sample_power(
            p1=p1,
            p2=p2,
            n1=sample_size_tensor,
            n2=sample_size_tensor,  # Equal sample sizes
            alpha=self.alpha,
            alternative=self.alternative,
        )
