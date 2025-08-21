from typing import Any, Optional

import numpy as np
import torch
from torchmetrics import Metric

import beignet


class TTestSampleSize(Metric):
    """TorchMetrics wrapper for one-sample and paired t-test sample size calculation.

    This metric computes the required sample size for one-sample t-tests or
    paired-samples t-tests to achieve a specified statistical power.

    Example:
        >>> metric = TTestSampleSize(power=0.8)
        >>> effect_sizes = torch.tensor([0.3, 0.5, 0.8])
        >>> metric.update(effect_sizes)
        >>> metric.compute()
        tensor([103., 42., 20.])
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        """Initialize the TTestOneSampleSampleSize metric.

        Args:
            power: Desired statistical power. Default: 0.8.
            alpha: Significance level (Type I error rate). Default: 0.05.
            alternative: Type of alternative hypothesis. Either "two-sided"
                or "one-sided". Default: "two-sided".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.power = power
        self.alpha = alpha
        self.alternative = alternative

        self.add_state("effect_sizes", default=[], dist_reduce_fx="cat")

    def update(self, effect_size: torch.Tensor) -> None:
        """Update the metric state with new effect sizes.

        Args:
            effect_size: Standardized effect size (Cohen's d).
        """
        self.effect_sizes.append(effect_size.detach())

    def compute(self) -> torch.Tensor:
        """Compute the required sample size for all accumulated effect sizes.

        Returns:
            Required sample sizes.
        """
        effect_sizes = torch.cat(self.effect_sizes, dim=0)

        return beignet.t_test_sample_size(
            effect_sizes,
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
        """Plot t-test sample size curves varying different parameters.

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
        >>> metric = TTestSampleSize()
        >>> fig = metric.plot(dep_var="effect_size", power=[0.7, 0.8, 0.9])
        >>> fig = metric.plot(dep_var="power", effect_size=[0.2, 0.5, 0.8])
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
                    sample_size_val = beignet.t_test_sample_size(
                        effect_size=torch.tensor(float(x_val)),
                        power=float(param_val),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "power":
                    sample_size_val = beignet.t_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        power=float(x_val),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "alpha":
                    sample_size_val = beignet.t_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        power=self.power,
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
            title = f"Sample Size Analysis: One-Sample T-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
