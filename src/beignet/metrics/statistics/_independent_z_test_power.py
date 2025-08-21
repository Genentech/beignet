"""Independent z-test power metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._independent_z_test_power import independent_z_test_power


class IndependentZTestPower(Metric):
    """Computes statistical power for independent samples z-tests with known variances.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two population means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled.
    sample_size1 : Tensor
        Sample size for group 1.
    sample_size2 : Tensor, optional
        Sample size for group 2. If None, assumed equal to sample_size1.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Examples
    --------
    >>> metric = NormalIndependentPower()
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size1 = torch.tensor(30)
    >>> sample_size2 = torch.tensor(30)
    >>> metric.update(effect_size, sample_size1, sample_size2)
    >>> metric.compute()
    tensor(0.1352)
    """

    full_state_update: bool = False

    def __init__(
        self, alpha: float = 0.05, alternative: str = "two-sided", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.alternative = alternative
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("sample_size1", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("sample_size2", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        effect_size: Tensor,
        sample_size1: Tensor,
        sample_size2: Tensor | None = None,
    ) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Standardized effect size (Cohen's d).
        sample_size1 : Tensor
            Sample size for group 1.
        sample_size2 : Tensor, optional
            Sample size for group 2.
        """
        if sample_size2 is None:
            sample_size2 = sample_size1

        if self.total == 0:
            self.effect_size = effect_size
            self.sample_size1 = sample_size1
            self.sample_size2 = sample_size2
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.sample_size1 = (self.sample_size1 * self.total + sample_size1) / (
                self.total + 1
            )
            self.sample_size2 = (self.sample_size2 * self.total + sample_size2) / (
                self.total + 1
            )
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the statistical power."""
        return independent_z_test_power(
            effect_size=self.effect_size,
            sample_size1=self.sample_size1,
            sample_size2=self.sample_size2,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        sample_size1: Optional[Any] = None,
        sample_size2: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "sample_size1", or "sample_size2".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        sample_size1 : array-like, optional
            Range of sample sizes for group 1 to plot. If None, uses reasonable default range.
        sample_size2 : array-like, optional
            Range of sample sizes for group 2 to plot. If None, uses reasonable default range.
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
        >>> metric = NormalIndependentPower()
        >>> fig = metric.plot(dep_var="effect_size", sample_size1=[20, 30, 40])
        >>> fig = metric.plot(dep_var="sample_size1", effect_size=[0.2, 0.5, 0.8])
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
        if dep_var == "effect_size":
            if effect_size is None:
                x_values = np.linspace(0.1, 1.5, 100)
            else:
                x_values = np.asarray(effect_size)

            if sample_size1 is None:
                param_values = [20, 30, 40]  # Different sample sizes
                param_name = "Sample Size 1"
                default_sample_size2 = None  # Will use equal sample sizes
            else:
                param_values = np.asarray(sample_size1)
                param_name = "Sample Size 1"
                default_sample_size2 = None

            x_label = "Effect Size (Cohen's d)"

        elif dep_var == "sample_size1":
            if sample_size1 is None:
                x_values = np.linspace(5, 100, 100)
            else:
                x_values = np.asarray(sample_size1)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]  # Small, medium, large effects
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Sample Size (Group 1)"

        elif dep_var == "sample_size2":
            if sample_size2 is None:
                x_values = np.linspace(5, 100, 100)
            else:
                x_values = np.asarray(sample_size2)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]
                param_name = "Effect Size"
                default_sample_size1 = 30
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_sample_size1 = 30

            x_label = "Sample Size (Group 2)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'sample_size1', or 'sample_size2', got {dep_var}"
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = independent_z_test_power(
                        effect_size=torch.tensor(float(x_val)),
                        sample_size1=torch.tensor(float(param_val)),
                        sample_size2=default_sample_size2,
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "sample_size1":
                    power_val = independent_z_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size1=torch.tensor(float(x_val)),
                        sample_size2=None,  # Equal sample sizes
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "sample_size2":
                    power_val = independent_z_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size1=torch.tensor(float(default_sample_size1)),
                        sample_size2=torch.tensor(float(x_val)),
                        alpha=alpha,
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
            title = f"Power Analysis: Independent Samples Z-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
