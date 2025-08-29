from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.statistics


class ProportionPower(Metric):
    """TorchMetrics wrapper for one-sample proportion power calculation.

    This metric accumulates proportions and sample sizes across batches,
    then computes the statistical power for detecting a difference from
    the null hypothesis proportion.

    Parameters
    ----------
    p0 : float
        Null hypothesis proportion (between 0 and 1).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Examples
    --------
    >>> metric = ProportionPower(p0=0.5, alpha=0.05)
    >>> metric.update(p1=torch.tensor([0.6, 0.7]), sample_size=torch.tensor([100, 150]))
    >>> metric.compute()
    tensor([0.7139, 0.9661])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        p0: float,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.p0 = p0
        self.alpha = alpha
        self.alternative = alternative

        self.add_state("p1_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_sizes", default=[], dist_reduce_fx="cat")

    def update(self, p1: Tensor, sample_size: Tensor) -> None:
        """Update the metric state with new proportion and sample size values.

        Parameters
        ----------
        p1 : Tensor
            Alternative hypothesis proportion(s).
        sample_size : Tensor
            Sample size(s) for the test.
        """
        self.p1_values.append(torch.atleast_1d(p1.detach()))
        self.sample_sizes.append(torch.atleast_1d(sample_size.detach()))

    def compute(self) -> Tensor:
        """Compute the statistical power for the accumulated data.

        Returns
        -------
        Tensor
            Statistical power values.
        """
        if not self.p1_values:
            raise RuntimeError("No values have been added to the metric.")

        p1_tensor = torch.cat(self.p1_values, dim=0)
        sample_size_tensor = torch.cat(self.sample_sizes, dim=0)
        p0_tensor = torch.full_like(p1_tensor, self.p0)

        return beignet.statistics.proportion_power(
            p0=p0_tensor,
            p1=p1_tensor,
            sample_size=sample_size_tensor,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "p1",
        p1: Optional[Any] = None,
        sample_size: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot proportion power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="p1"
            The variable to plot on x-axis. Either "p1" or "sample_size".
        p1 : array-like, optional
            Range of alternative proportions to plot. If None, uses reasonable default range.
        sample_size : array-like, optional
            Range of sample sizes to plot. If None, uses reasonable default range.
        alpha : float, default=0.05
            Significance level for power calculation.
        ax : matplotlib Axes, optional
            Axes object to plot on. If None, creates new figure.
        title : str, optional
            Plot title. If None, generates automatic title.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot function.

        Returns
        -------
        fig : matplotlib Figure
            The figure object containing the plot.

        Examples
        --------
        >>> metric = ProportionPower(p0=0.5)
        >>> fig = metric.plot(dep_var="p1", sample_size=[50, 100, 200])
        >>> fig = metric.plot(dep_var="sample_size", p1=[0.3, 0.6, 0.7])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from err

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # Set default ranges based on dep_var
        if dep_var == "p1":
            if p1 is None:
                # Center range around p0 with reasonable spread
                x_values = np.linspace(
                    max(0.01, self.p0 - 0.4), min(0.99, self.p0 + 0.4), 100
                )
            else:
                x_values = np.asarray(p1)

            if sample_size is None:
                param_values = [50, 100, 200]  # Different sample sizes
                param_name = "Sample Size"
            else:
                param_values = np.asarray(sample_size)
                param_name = "Sample Size"

            x_label = "Alternative Proportion (p1)"

        elif dep_var == "sample_size":
            if sample_size is None:
                x_values = np.linspace(20, 500, 100)
            else:
                x_values = np.asarray(sample_size)

            if p1 is None:
                # Show proportions at reasonable distance from p0
                if self.p0 <= 0.5:
                    param_values = [self.p0 + 0.1, self.p0 + 0.2, self.p0 + 0.3]
                else:
                    param_values = [self.p0 - 0.1, self.p0 - 0.2, self.p0 - 0.3]
                param_name = "p1"
            else:
                param_values = np.asarray(p1)
                param_name = "p1"

            x_label = "Sample Size"
        else:
            raise ValueError(f"dep_var must be 'p1' or 'sample_size', got {dep_var}")

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "p1":
                    power_val = beignet.statistics.proportion_power(
                        p0=torch.tensor(float(self.p0)),
                        p1=torch.tensor(float(x_val)),
                        sample_size=torch.tensor(float(param_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "sample_size":
                    power_val = beignet.statistics.proportion_power(
                        p0=torch.tensor(float(self.p0)),
                        p1=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(x_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )

                y_values.append(float(power_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Add vertical line for null hypothesis if plotting p1
        if dep_var == "p1":
            ax.axvline(
                x=self.p0,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"p0 = {self.p0}",
            )

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = f"Power Analysis: One-Sample Proportion Test (p0={self.p0}, {self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
