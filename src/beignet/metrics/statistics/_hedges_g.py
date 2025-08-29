from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class HedgesG(Metric):
    r"""
    Compute Hedges' g effect size metric between two groups.

    This metric calculates Hedges' g, a bias-corrected version of Cohen's d
    that provides more accurate effect size estimates, especially for small
    sample sizes. It follows TorchMetrics conventions for stateful metric
    computation.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import HedgesG
    >>> metric = HedgesG()
    >>> group1 = torch.randn(10)
    >>> group2 = torch.randn(10) + 0.5  # Add some effect
    >>> metric.update(group1, group2)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # State for accumulating samples
        self.add_state("group1_samples", default=[], dist_reduce_fx="cat")
        self.add_state("group2_samples", default=[], dist_reduce_fx="cat")

    def update(self, group1: Tensor, group2: Tensor) -> None:
        """
        Update the metric state with new samples.

        Parameters
        ----------
        group1 : Tensor
            Samples from the first group.
        group2 : Tensor
            Samples from the second group.
        """
        self.group1_samples.append(group1)
        self.group2_samples.append(group2)

    def compute(self) -> Tensor:
        """
        Compute Hedges' g based on accumulated samples.

        Returns
        -------
        Tensor
            The computed Hedges' g effect size.
        """
        if not self.group1_samples or not self.group2_samples:
            raise RuntimeError("No samples have been added to the metric.")

        # Concatenate all samples
        group1_all = torch.cat(self.group1_samples, dim=-1)
        group2_all = torch.cat(self.group2_samples, dim=-1)

        # Inline implementation of hedges_g to avoid circular imports

        # First compute Cohen's d (inline implementation)
        mean1 = torch.mean(group1_all, dim=-1)
        mean2 = torch.mean(group2_all, dim=-1)

        # Pooled standard deviation
        n1 = group1_all.shape[-1]
        n2 = group2_all.shape[-1]

        var1 = torch.var(group1_all, dim=-1, unbiased=True)
        var2 = torch.var(group2_all, dim=-1, unbiased=True)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = torch.sqrt(pooled_var)

        cohens_d_value = (mean1 - mean2) / pooled_std

        # Now apply Hedges' g correction
        # Degrees of freedom
        df = n1 + n2 - 2

        # Bias correction factor (Hedges & Olkin, 1985)
        # J(df) = Gamma((df+1)/2) / (sqrt(df/2) * Gamma(df/2))
        # Approximation: J(df) ≈ 1 - 3/(4*df - 1)
        correction_factor = 1.0 - 3.0 / (4.0 * df - 1.0)

        # Apply correction
        return cohens_d_value * correction_factor

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.group1_samples = []
        self.group2_samples = []

    def plot(
        self,
        plot_type: str = "distribution",
        group_names: Optional[list] = None,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Hedges' g visualization comparing two groups.

        Parameters
        ----------
        plot_type : str, default="distribution"
            Type of plot to create. Options are "distribution" or "effect_size".
        group_names : list, optional
            Names for the two groups. If None, uses default names.
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
        >>> from beignet.metrics import HedgesG
        >>> metric = HedgesG()
        >>> group1 = torch.randn(100)
        >>> group2 = torch.randn(100) + 0.5
        >>> metric.update(group1, group2)
        >>> fig = metric.plot(plot_type="distribution")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.group1_samples or not self.group2_samples:
            raise RuntimeError(
                "No samples have been added to the metric. Call update() first.",
            )

        # Get the accumulated data
        group1_all = torch.cat(self.group1_samples, dim=-1)
        group2_all = torch.cat(self.group2_samples, dim=-1)

        # Compute Hedges' g
        effect_size = self.compute()

        # Set default group names
        if group_names is None:
            group_names = ["Group 1", "Group 2"]

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "distribution":
            # Plot histograms of both groups
            group1_np = group1_all.detach().cpu().numpy()
            group2_np = group2_all.detach().cpu().numpy()

            ax.hist(group1_np, alpha=0.7, label=group_names[0], bins=30, **kwargs)
            ax.hist(group2_np, alpha=0.7, label=group_names[1], bins=30, **kwargs)

            # Add vertical lines for means
            ax.axvline(
                group1_np.mean(),
                color="blue",
                linestyle="--",
                alpha=0.8,
                label=f"{group_names[0]} mean",
            )
            ax.axvline(
                group2_np.mean(),
                color="orange",
                linestyle="--",
                alpha=0.8,
                label=f"{group_names[1]} mean",
            )

            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()

            if title is None:
                title = (
                    f"Distribution Comparison (Hedges' g = {float(effect_size):.3f})"
                )

        elif plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = [f"{group_names[0]}\nvs\n{group_names[1]}"]
            effect_sizes = [float(effect_size)]

            bars = ax.bar(categories, effect_sizes, **kwargs)

            # Color code by effect size magnitude (same thresholds as Cohen's d)
            color = "lightblue"
            if abs(effect_sizes[0]) >= 0.8:
                color = "red"  # Large effect
            elif abs(effect_sizes[0]) >= 0.5:
                color = "orange"  # Medium effect
            elif abs(effect_sizes[0]) >= 0.2:
                color = "yellow"  # Small effect

            bars[0].set_color(color)

            # Add reference lines for effect size thresholds
            ax.axhline(
                y=0.2,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Small (0.2)",
            )
            ax.axhline(
                y=0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Medium (0.5)",
            )
            ax.axhline(
                y=0.8,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label="Large (0.8)",
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_ylabel("Hedges' g")
            ax.legend()

            if title is None:
                title = f"Effect Size: Hedges' g = {float(effect_size):.3f}"
        else:
            raise ValueError(
                f"plot_type must be 'distribution' or 'effect_size', got {plot_type}",
            )

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
