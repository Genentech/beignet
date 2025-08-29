"""Z-test power metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.statistics


class ZTestPower(Metric):
    """Computes statistical power for one-sample z-tests with known variance.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the true population mean and the hypothesized mean, divided by the
        population standard deviation: d = (μ₁ - μ₀) / σ.
    sample_size : Tensor
        Sample size (number of observations).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Examples
    --------
    >>> metric = ZTestPower()
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size = torch.tensor(30)
    >>> metric.update(effect_size, sample_size)
    >>> metric.compute()
    tensor(0.1905)
    """

    full_state_update: bool = False

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.alternative = alternative
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("sample_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, sample_size: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Standardized effect size (Cohen's d).
        sample_size : Tensor
            Sample size.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.sample_size = sample_size
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.sample_size = (self.sample_size * self.total + sample_size) / (
                self.total + 1
            )
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the statistical power."""
        return beignet.statistics.z_test_power(
            input=self.effect_size,
            sample_size=self.sample_size,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "sample_size",
        sample_size: Optional[Any] = None,
        effect_size: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="sample_size"
            The variable to plot on x-axis. Either "sample_size", "effect_size", or "alpha".
        sample_size : array-like, optional
            Range of sample sizes to plot. If None, uses reasonable default range.
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
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
        >>> metric = ZTestPower()
        >>> fig = metric.plot(dep_var="sample_size", effect_size=[0.2, 0.5, 0.8])
        >>> fig = metric.plot(dep_var="effect_size", sample_size=[10, 30, 50])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from None

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # Set default ranges based on dep_var
        if dep_var == "sample_size":
            if sample_size is None:
                x_values = np.linspace(5, 100, 100)
            else:
                x_values = np.asarray(sample_size)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]  # Small, medium, large effects
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Sample Size"

        elif dep_var == "effect_size":
            if effect_size is None:
                x_values = np.linspace(0.1, 1.5, 100)
            else:
                x_values = np.asarray(effect_size)

            if sample_size is None:
                param_values = [10, 30, 50]  # Different sample sizes
                param_name = "Sample Size"
            else:
                param_values = np.asarray(sample_size)
                param_name = "Sample Size"

            x_label = "Effect Size (Cohen's d)"

        elif dep_var == "alpha":
            if alpha is None:
                x_values = np.linspace(0.01, 0.10, 100)
            else:
                x_values = np.asarray(alpha)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Significance Level (α)"
        else:
            raise ValueError(
                f"dep_var must be 'sample_size', 'effect_size', or 'alpha', got {dep_var}",
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "sample_size":
                    power_val = beignet.statistics.z_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(x_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "effect_size":
                    power_val = beignet.statistics.z_test_power(
                        effect_size=torch.tensor(float(x_val)),
                        sample_size=torch.tensor(float(param_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "alpha":
                    power_val = beignet.statistics.z_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(30.0),  # Default sample size
                        alpha=float(x_val),
                        alternative=self.alternative,
                    )

                y_values.append(float(power_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = f"Power Analysis: One-Sample Z-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
