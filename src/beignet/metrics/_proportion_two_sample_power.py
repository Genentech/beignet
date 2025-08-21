import torch
from torch import Tensor
from torchmetrics import Metric

import beignet


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

        self.add_state("p1_values", default=[], dist_reduce_fx="cat")
        self.add_state("p2_values", default=[], dist_reduce_fx="cat")
        self.add_state("n1_values", default=[], dist_reduce_fx="cat")
        self.add_state("n2_values", default=[], dist_reduce_fx="cat")

    def update(
        self, p1: Tensor, p2: Tensor, n1: Tensor, n2: Tensor | None = None
    ) -> None:
        """Update the metric state with new proportion and sample size values.

        Parameters
        ----------
        p1 : Tensor
            Proportion in group 1.
        p2 : Tensor
            Proportion in group 2.
        n1 : Tensor
            Sample size for group 1.
        n2 : Tensor, optional
            Sample size for group 2. If None, assumes equal sample sizes.
        """
        if n2 is None:
            n2 = n1

        self.p1_values.append(p1.detach())
        self.p2_values.append(p2.detach())
        self.n1_values.append(n1.detach())
        self.n2_values.append(n2.detach())

    def compute(self) -> Tensor:
        """Compute the statistical power for the accumulated data.

        Returns
        -------
        Tensor
            Statistical power values.
        """
        if not self.p1_values:
            return torch.tensor([], dtype=torch.float32)

        p1_tensor = torch.cat(self.p1_values, dim=0)
        p2_tensor = torch.cat(self.p2_values, dim=0)
        n1_tensor = torch.cat(self.n1_values, dim=0)
        n2_tensor = torch.cat(self.n2_values, dim=0)

        return beignet.proportion_two_sample_power(
            p1=p1_tensor,
            p2=p2_tensor,
            n1=n1_tensor,
            n2=n2_tensor,
            alpha=self.alpha,
            alternative=self.alternative,
        )
