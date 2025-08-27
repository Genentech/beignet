from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet


class AnalysisOfVariancePower(Metric):
    """TorchMetrics wrapper for Analysis of Variance power calculation.

    This metric accumulates effect sizes, sample sizes, and number of groups across batches,
    then computes the statistical power for one-way Analysis of Variance F-tests.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).
    **kwargs
        Additional keyword arguments passed to the parent Metric class.

    Examples
    --------
    >>> metric = AnalysisOfVariancePower(alpha=0.05)
    >>> effect_size = torch.tensor([0.25, 0.40])
    >>> sample_size = torch.tensor([120, 90])
    >>> k = torch.tensor([3, 4])
    >>> metric.update(effect_size=effect_size, sample_size=sample_size, k=k)
    >>> metric.compute()
    tensor([0.7061, 0.8234])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, alpha: float = 0.05, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha

        self.add_state("effect_size_list", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_list", default=[], dist_reduce_fx="cat")
        self.add_state("k_list", default=[], dist_reduce_fx="cat")

    def update(self, effect_size: Tensor, sample_size: Tensor, k: Tensor) -> None:
        """Update the metric state with new effect sizes, sample sizes, and group counts.

        Parameters
        ----------
        effect_size : Tensor
            Cohen's f effect size.
        sample_size : Tensor
            Total sample size across all groups.
        k : Tensor
            Number of groups in the Analysis of Variance.
        """
        self.effect_size_list.append(effect_size.detach())
        self.sample_size_list.append(sample_size.detach())
        self.k_list.append(k.detach())

    def compute(self) -> Tensor:
        """Compute the statistical power for the accumulated data.

        Returns
        -------
        Tensor
            Statistical power values.
        """
        if not self.effect_size_list:
            return torch.tensor([], dtype=torch.float32)

        effect_size_tensor = torch.cat(self.effect_size_list, dim=0)
        sample_size_tensor = torch.cat(self.sample_size_list, dim=0)
        k_tensor = torch.cat(self.k_list, dim=0)

        return beignet.analysis_of_variance_power(
            effect_size=effect_size_tensor,
            sample_size=sample_size_tensor,
            k=k_tensor,
            alpha=self.alpha,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        sample_size: Optional[Any] = None,
        k: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Analysis of Variance power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "sample_size", or "k".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        sample_size : array-like, optional
            Range of sample sizes to plot. If None, uses reasonable default range.
        k : array-like, optional
            Range of number of groups to plot. If None, uses reasonable default range.
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
        >>> metric = AnalysisOfVariancePower()
        >>> fig = metric.plot(dep_var="effect_size", k=[3, 4, 5])
        >>> fig = metric.plot(dep_var="sample_size", effect_size=[0.1, 0.25, 0.4])
        >>> fig = metric.plot(dep_var="k", sample_size=[30, 60, 90])
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
                x_values = np.linspace(0.1, 0.8, 100)
            else:
                x_values = np.asarray(effect_size)

            if k is None:
                param_values = [3, 4, 5]  # Different number of groups
                param_name = "k (groups)"
                default_sample_size = 60
            else:
                param_values = np.asarray(k)
                param_name = "k (groups)"
                default_sample_size = 60

            x_label = "Effect Size (Cohen's f)"

        elif dep_var == "sample_size":
            if sample_size is None:
                x_values = np.linspace(20, 200, 100)
            else:
                x_values = np.asarray(sample_size)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]  # Small, medium, large effects
                param_name = "Effect Size"
                default_k = 3
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_k = 3

            x_label = "Sample Size"

        elif dep_var == "k":
            if k is None:
                x_values = np.linspace(2, 10, 50)
            else:
                x_values = np.asarray(k)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]
                param_name = "Effect Size"
                default_sample_size = 60
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_sample_size = 60

            x_label = "Number of Groups (k)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'sample_size', or 'k', got {dep_var}",
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = beignet.analysis_of_variance_power(
                        effect_size=torch.tensor(float(x_val)),
                        sample_size=torch.tensor(float(default_sample_size)),
                        k=torch.tensor(int(param_val)),
                        alpha=alpha,
                    )
                elif dep_var == "sample_size":
                    power_val = beignet.analysis_of_variance_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(x_val)),
                        k=torch.tensor(int(default_k)),
                        alpha=alpha,
                    )
                elif dep_var == "k":
                    power_val = beignet.analysis_of_variance_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(default_sample_size)),
                        k=torch.tensor(int(x_val)),
                        alpha=alpha,
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
            title = "Power Analysis: One-Way Analysis of Variance"
        ax.set_title(title)

        plt.tight_layout()
        return fig
