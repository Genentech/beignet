from typing import Any, Optional

import numpy as np
import torch
from torchmetrics import Metric

import beignet.statistics


class IndependentTTestSampleSize(Metric):
    """TorchMetrics wrapper for independent samples t-test sample size calculation.

    This metric computes the required sample size for group 1 in independent
    samples t-tests to achieve a specified statistical power.

    Example:
        >>> metric = IndependentTTestSampleSize(power=0.8)
        >>> effect_sizes = torch.tensor([0.3, 0.5, 0.8])
        >>> metric.update(effect_sizes)
        >>> metric.compute()
        tensor([176., 78., 38.])
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        ratio: float = 1.0,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        """Initialize the TTestIndependentSampleSize metric.

        Args:
            ratio: Ratio of sample size for group 2 relative to group 1. Default: 1.0.
            power: Desired statistical power. Default: 0.8.
            alpha: Significance level (Type I error rate). Default: 0.05.
            alternative: Type of alternative hypothesis. Either "two-sided",
                "larger", or "smaller". Default: "two-sided".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.ratio = ratio
        self.power = power
        self.alpha = alpha
        self.alternative = alternative

        self.add_state("effect_sizes", default=[], dist_reduce_fx="cat")
        self.add_state("ratio_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: torch.Tensor,
        ratio: torch.Tensor | None = None,
    ) -> None:
        """Update the metric state with new values.

        Args:
            effect_size: Standardized effect size (Cohen's d).
            ratio: Ratio of sample size for group 2 relative to group 1.
                If None, uses the default ratio from initialization.
        """
        self.effect_sizes.append(torch.atleast_1d(effect_size.detach()))

        if ratio is None:
            ratio = torch.full_like(effect_size, self.ratio)
        self.ratio_values.append(torch.atleast_1d(ratio.detach()))

    def compute(self) -> torch.Tensor:
        """Compute the required sample size for all accumulated values.

        Returns:
            Required sample sizes for group 1.
        """
        effect_sizes = torch.cat(self.effect_sizes, dim=0)
        ratio_values = torch.cat(self.ratio_values, dim=0)

        if not self.effect_sizes:
            raise RuntimeError("No values have been added to the metric.")

        return beignet.statistics.independent_t_test_sample_size(
            input=effect_sizes,
            ratio=ratio_values,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        ratio: Optional[Any] = None,
        power: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot independent t-test sample size curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "ratio", or "power".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        ratio : array-like, optional
            Range of sample size ratios to plot. If None, uses reasonable default range.
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
        >>> metric = IndependentTTestSampleSize()
        >>> fig = metric.plot(dep_var="effect_size", power=[0.7, 0.8, 0.9])
        >>> fig = metric.plot(dep_var="power", effect_size=[0.2, 0.5, 0.8])
        >>> fig = metric.plot(dep_var="ratio", effect_size=[0.5])
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

        elif dep_var == "ratio":
            if ratio is None:
                x_values = np.linspace(0.5, 3.0, 100)
            else:
                x_values = np.asarray(ratio)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Sample Size Ratio (n2/n1)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'ratio', or 'power', got {dep_var}",
            )

        # Plot sample size curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    sample_size_val = beignet.statistics.independent_t_test_sample_size(
                        input=torch.tensor(float(x_val)),
                        ratio=torch.tensor(float(self.ratio)),
                        power=float(param_val),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "power":
                    sample_size_val = beignet.statistics.independent_t_test_sample_size(
                        input=torch.tensor(float(param_val)),
                        ratio=torch.tensor(float(self.ratio)),
                        power=float(x_val),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "ratio":
                    sample_size_val = beignet.statistics.independent_t_test_sample_size(
                        input=torch.tensor(float(param_val)),
                        ratio=torch.tensor(float(x_val)),
                        power=self.power,
                        alpha=alpha,
                        alternative=self.alternative,
                    )

                y_values.append(float(sample_size_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Required Sample Size (Group 1)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = f"Sample Size Analysis: Independent T-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
