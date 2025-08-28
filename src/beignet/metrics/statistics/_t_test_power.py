from typing import Any, Optional

import numpy as np
import torch
from torchmetrics import Metric

import beignet.statistics

from ..functional.statistics import t_test_power


class TTestPower(Metric):
    """TorchMetrics wrapper for one-sample and paired t-test power calculation.

    This metric computes the statistical power for one-sample t-tests or
    paired-samples t-tests, which is the probability of correctly rejecting
    a false null hypothesis.

    Example:
        >>> metric = TTestPower()
        >>> effect_sizes = torch.tensor([0.3, 0.5, 0.8])
        >>> sample_sizes = torch.tensor([20, 30, 40])
        >>> metric.update(effect_sizes, sample_sizes)
        >>> metric.compute()
        tensor([0.3684, 0.6595, 0.9422])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        """Initialize the TTestOneSamplePower metric.

        Args:
            alpha: Significance level (Type I error rate). Default: 0.05.
            alternative: Type of alternative hypothesis. Either "two-sided"
                or "one-sided". Default: "two-sided".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.alternative = alternative

        # Validate parameters
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(
                f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
            )

        self.add_state("group1_samples", default=[], dist_reduce_fx="cat")
        self.add_state("group2_samples", default=[], dist_reduce_fx="cat")

    def update(self, group1: torch.Tensor, group2: torch.Tensor) -> None:
        """Update the metric state with new samples.

        Args:
            group1: Samples from the first group.
            group2: Samples from the second group.
        """
        self.group1_samples.append(group1.detach())
        self.group2_samples.append(group2.detach())

    def compute(self) -> torch.Tensor:
        """Compute the statistical power for all accumulated values.

        Returns:
            Statistical power values.
        """
        if not self.group1_samples or not self.group2_samples:
            raise RuntimeError("No samples have been added to the metric.")

        # Concatenate all samples
        group1_all = torch.cat(self.group1_samples, dim=0)
        group2_all = torch.cat(self.group2_samples, dim=0)

        # Use functional implementation
        return t_test_power(
            group1_all,
            group2_all,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        sample_size: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot t-test power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "sample_size", or "alpha".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
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
        >>> metric = TTestPower()
        >>> fig = metric.plot(dep_var="effect_size", sample_size=[10, 20, 30])
        >>> fig = metric.plot(dep_var="sample_size", effect_size=[0.2, 0.5, 0.8])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
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

            if sample_size is None:
                param_values = [10, 20, 30]  # Different sample sizes
                param_name = "Sample Size"
            else:
                param_values = np.asarray(sample_size)
                param_name = "Sample Size"

            x_label = "Effect Size (Cohen's d)"

        elif dep_var == "sample_size":
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

        elif dep_var == "alpha":
            if alpha is None:
                x_values = np.linspace(0.01, 0.10, 100)
            else:
                x_values = np.asarray(alpha)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]
                param_name = "Effect Size"
                default_sample_size = 30
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_sample_size = 30

            x_label = "Significance Level (α)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'sample_size', or 'alpha', got {dep_var}",
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = beignet.statistics.t_test_power(
                        effect_size=torch.tensor(float(x_val)),
                        sample_size=torch.tensor(float(param_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "sample_size":
                    power_val = beignet.statistics.t_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(x_val)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "alpha":
                    power_val = beignet.statistics.t_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(default_sample_size)),
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
            title = f"Power Analysis: One-Sample T-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, alternative='{self.alternative}')"
