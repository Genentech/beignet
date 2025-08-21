from typing import Any, Optional

import numpy as np
import torch
from torchmetrics import Metric

import beignet


class IndependentTTestPower(Metric):
    """TorchMetrics wrapper for independent samples t-test power calculation.

    This metric computes the statistical power for independent samples t-tests
    (two-sample t-tests), which is the probability of correctly rejecting
    a false null hypothesis.

    Example:
        >>> metric = IndependentTTestPower()
        >>> effect_sizes = torch.tensor([0.3, 0.5, 0.8])
        >>> nobs1 = torch.tensor([30, 30, 30])
        >>> metric.update(effect_sizes, nobs1)
        >>> metric.compute()
        tensor([0.1793, 0.3823, 0.7190])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        ratio: float = 1.0,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        """Initialize the TTestIndependentPower metric.

        Args:
            ratio: Ratio of sample size for group 2 relative to group 1. Default: 1.0.
            alpha: Significance level (Type I error rate). Default: 0.05.
            alternative: Type of alternative hypothesis. Either "two-sided",
                "larger", or "smaller". Default: "two-sided".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.ratio = ratio
        self.alpha = alpha
        self.alternative = alternative

        self.add_state("effect_sizes", default=[], dist_reduce_fx="cat")
        self.add_state("nobs1_values", default=[], dist_reduce_fx="cat")
        self.add_state("ratio_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: torch.Tensor,
        nobs1: torch.Tensor,
        ratio: torch.Tensor | None = None,
    ) -> None:
        """Update the metric state with new values.

        Args:
            effect_size: Standardized effect size (Cohen's d).
            nobs1: Number of observations in group 1.
            ratio: Ratio of sample size for group 2 relative to group 1.
                If None, uses the default ratio from initialization.
        """
        self.effect_sizes.append(effect_size.detach())
        self.nobs1_values.append(nobs1.detach())

        if ratio is None:
            ratio = torch.full_like(effect_size, self.ratio)
        self.ratio_values.append(ratio.detach())

    def compute(self) -> torch.Tensor:
        """Compute the statistical power for all accumulated values.

        Returns:
            Statistical power values.
        """
        effect_sizes = torch.cat(self.effect_sizes, dim=0)
        nobs1_values = torch.cat(self.nobs1_values, dim=0)
        ratio_values = torch.cat(self.ratio_values, dim=0)

        return beignet.independent_t_test_power(
            effect_sizes,
            nobs1_values,
            ratio_values,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        nobs1: Optional[Any] = None,
        ratio: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot independent t-test power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "nobs1", or "ratio".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        nobs1 : array-like, optional
            Range of sample sizes (group 1) to plot. If None, uses reasonable default range.
        ratio : array-like, optional
            Range of sample size ratios to plot. If None, uses reasonable default range.
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
        >>> metric = IndependentTTestPower()
        >>> fig = metric.plot(dep_var="effect_size", nobs1=[20, 30, 40])
        >>> fig = metric.plot(dep_var="nobs1", effect_size=[0.2, 0.5, 0.8])
        >>> fig = metric.plot(dep_var="ratio", effect_size=[0.5], nobs1=[30])
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

            if nobs1 is None:
                param_values = [20, 30, 40]  # Different sample sizes
                param_name = "n1 (Group 1)"
                default_ratio = self.ratio
            else:
                param_values = np.asarray(nobs1)
                param_name = "n1 (Group 1)"
                default_ratio = self.ratio

            x_label = "Effect Size (Cohen's d)"

        elif dep_var == "nobs1":
            if nobs1 is None:
                x_values = np.linspace(5, 100, 100)
            else:
                x_values = np.asarray(nobs1)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]  # Small, medium, large effects
                param_name = "Effect Size"
                default_ratio = self.ratio
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_ratio = self.ratio

            x_label = "Sample Size (Group 1)"

        elif dep_var == "ratio":
            if ratio is None:
                x_values = np.linspace(0.5, 3.0, 100)
            else:
                x_values = np.asarray(ratio)

            if effect_size is None:
                param_values = [0.2, 0.5, 0.8]
                param_name = "Effect Size"
                default_nobs1 = 30
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_nobs1 = 30

            x_label = "Sample Size Ratio (n2/n1)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'nobs1', or 'ratio', got {dep_var}"
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = beignet.independent_t_test_power(
                        effect_size=torch.tensor(float(x_val)),
                        nobs1=torch.tensor(float(param_val)),
                        ratio=torch.tensor(float(default_ratio)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "nobs1":
                    power_val = beignet.independent_t_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        nobs1=torch.tensor(float(x_val)),
                        ratio=torch.tensor(float(default_ratio)),
                        alpha=alpha,
                        alternative=self.alternative,
                    )
                elif dep_var == "ratio":
                    power_val = beignet.independent_t_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        nobs1=torch.tensor(float(default_nobs1)),
                        ratio=torch.tensor(float(x_val)),
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
            title = f"Power Analysis: Independent T-Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig
