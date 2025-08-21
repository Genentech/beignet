"""Z-test sample size metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from beignet._z_test_sample_size import z_test_sample_size


class ZTestSampleSize(Metric):
    """Computes required sample size for one-sample z-tests with known variance.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the true population mean and the hypothesized mean, divided by the
        population standard deviation: d = (μ₁ - μ₀) / σ.
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Examples
    --------
    >>> metric = ZTestSampleSize()
    >>> effect_size = torch.tensor(0.5)
    >>> metric.update(effect_size)
    >>> metric.compute()
    tensor(50)
    """

    full_state_update: bool = False

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.power = power
        self.alpha = alpha
        self.alternative = alternative
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Standardized effect size (Cohen's d).
        """
        if self.total == 0:
            self.effect_size = effect_size
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the required sample size."""
        return z_test_sample_size(
            effect_size=self.effect_size,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        power: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot sample size curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "power", or "alpha".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        power : array-like, optional
            Range of power values to plot. If None, uses reasonable default range.
        alpha : float, default=0.05
            Significance level for sample size calculation.
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
        >>> metric = ZTestSampleSize()
        >>> fig = metric.plot(dep_var="effect_size", power=[0.7, 0.8, 0.9])
        >>> fig = metric.plot(dep_var="power", effect_size=[0.2, 0.5, 0.8])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from None

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # Set default ranges based on dep_var
        if dep_var == "effect_size":
            if effect_size is None:
                x_values = np.linspace(0.1, 1.5, 100)
            else:
                x_values = np.asarray(effect_size)

            if power is None:
                param_values = [0.7, 0.8, 0.9]  # Different power levels
                param_name = "Power"
            else:
                param_values = np.asarray(power)
                param_name = "Power"

            x_label = "Effect Size (Cohen's d)"

        elif dep_var == "power":
            if power is None:
                x_values = np.linspace(0.5, 0.95, 100)
            else:
                x_values = np.asarray(power)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]  # Small, medium, large effects
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Statistical Power"

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
                f"dep_var must be 'effect_size', 'power', or 'alpha', got {dep_var}"
            )

        # Plot sample size curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    sample_size_val = z_test_sample_size(
                        effect_size=torch.tensor(float(x_val)),
                        power=torch.tensor(float(param_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "power":
                    sample_size_val = z_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        power=torch.tensor(float(x_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "alpha":
                    sample_size_val = z_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        power=torch.tensor(0.8),  # Default power
                        alpha=float(x_val),
                        alternative=self.alternative,
                    )

                y_values.append(float(sample_size_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Required Sample Size")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = f"Sample Size Analysis: One-Sample Z-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
