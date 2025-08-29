from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.statistics


class CohensFSquared(Metric):
    """TorchMetrics wrapper for Cohen's f² effect size calculation.

    This metric accumulates group means and pooled standard deviations across batches,
    then computes Cohen's f² effect size for ANOVA.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to the parent Metric class.

    Examples
    --------
    >>> metric = CohensFSquared()
    >>> group_means = torch.tensor([[10.0, 12.0, 14.0], [5.0, 7.0, 9.0]])
    >>> pooled_std = torch.tensor([2.0, 1.5])
    >>> metric.update(group_means=group_means, pooled_std=pooled_std)
    >>> metric.compute()
    tensor([1.0000, 1.7778])
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("group_means_list", default=[], dist_reduce_fx="cat")
        self.add_state("pooled_std_list", default=[], dist_reduce_fx="cat")

    def update(self, group_means: Tensor, pooled_std: Tensor) -> None:
        """Update the metric state with new group means and pooled standard deviations.

        Parameters
        ----------
        group_means : Tensor, shape=(..., k)
            Means for each group. The last dimension represents different groups.
        pooled_std : Tensor
            Pooled within-group standard deviation.
        """
        self.group_means_list.append(torch.atleast_1d(group_means.detach()))
        self.pooled_std_list.append(torch.atleast_1d(pooled_std.detach()))

    def compute(self) -> Tensor:
        """Compute Cohen's f² effect size for the accumulated data.

        Returns
        -------
        Tensor
            Cohen's f² effect size values.
        """
        if not self.group_means_list:
            raise RuntimeError("No values have been added to the metric.")

        group_means_tensor = torch.cat(self.group_means_list, dim=0)
        pooled_std_tensor = torch.cat(self.pooled_std_list, dim=0)

        return beignet.statistics.cohens_f_squared(
            input=group_means_tensor,
            other=pooled_std_tensor,
        )

    def plot(
        self,
        plot_type: str = "effect_size",
        group_names: Optional[list] = None,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Cohen's f² effect size visualization.

        Parameters
        ----------
        plot_type : str, default="effect_size"
            Type of plot to create. Options are "means" or "effect_size".
        group_names : list, optional
            Names for the groups. If None, uses default names.
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
        >>> import torch
        >>> from beignet.metrics import CohensFSquared
        >>> metric = CohensFSquared()
        >>> group_means = torch.tensor([[10.0, 12.0, 14.0]])
        >>> pooled_std = torch.tensor([2.0])
        >>> metric.update(group_means, pooled_std)
        >>> fig = metric.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.group_means_list or not self.pooled_std_list:
            raise RuntimeError(
                "No samples have been added to the metric. Call update() first.",
            )

        # Get the accumulated data
        group_means_tensor = torch.cat(self.group_means_list, dim=0)
        pooled_std_tensor = torch.cat(self.pooled_std_list, dim=0)

        # Use the most recent/average data for plotting
        if group_means_tensor.dim() > 1:
            group_means = group_means_tensor[-1]  # Use most recent
            pooled_std = pooled_std_tensor[-1]
        else:
            group_means = group_means_tensor
            pooled_std = pooled_std_tensor

        # Compute Cohen's f²
        effect_size = self.compute()
        if effect_size.dim() > 0:
            effect_size = effect_size[-1]  # Use most recent

        n_groups = group_means.size(-1) if group_means.dim() > 0 else len(group_means)

        # Set default group names
        if group_names is None:
            group_names = [f"Group {i + 1}" for i in range(n_groups)]

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "means":
            # Plot group means with error bars
            means_np = group_means.detach().cpu().numpy()
            errors = [float(pooled_std)] * n_groups  # Same pooled std for all groups

            bars = ax.bar(group_names, means_np, yerr=errors, capsize=5, **kwargs)

            ax.set_xlabel("Groups")
            ax.set_ylabel("Mean Value")

            if title is None:
                title = (
                    f"Group Means Comparison (Cohen's f² = {float(effect_size):.3f})"
                )

        elif plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = ["Cohen's f²"]
            effect_sizes = [float(effect_size)]

            bars = ax.bar(categories, effect_sizes, **kwargs)

            # Color code by effect size magnitude (Cohen's f² conventions)
            color = "lightblue"
            if abs(effect_sizes[0]) >= 0.35:
                color = "red"  # Large effect (f²≥0.35, equivalent to f≥0.4)
            elif abs(effect_sizes[0]) >= 0.15:
                color = "orange"  # Medium effect (f²≥0.15, equivalent to f≥0.25)
            elif abs(effect_sizes[0]) >= 0.02:
                color = "yellow"  # Small effect (f²≥0.02, equivalent to f≥0.1)

            bars[0].set_color(color)

            # Add reference lines for effect size thresholds
            ax.axhline(
                y=0.02,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Small (0.02)",
            )
            ax.axhline(
                y=0.15,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Medium (0.15)",
            )
            ax.axhline(
                y=0.35,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label="Large (0.35)",
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_ylabel("Cohen's f²")
            ax.legend()

            if title is None:
                title = f"Effect Size: Cohen's f² = {float(effect_size):.3f}"
        else:
            raise ValueError(
                f"plot_type must be 'means' or 'effect_size', got {plot_type}",
            )

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
