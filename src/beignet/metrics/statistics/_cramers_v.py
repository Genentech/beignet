from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class CramersV(Metric):
    r"""
    Compute Cramer's V effect size for chi-square tests.

    This metric calculates Cramer's V, a measure of association between
    two nominal variables based on chi-square statistics. It follows
    TorchMetrics conventions for stateful metric computation.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import CramersV
    >>> metric = CramersV()
    >>> chi_sq = torch.tensor(10.5)
    >>> n = torch.tensor(100)
    >>> min_dim = torch.tensor(1)  # For a 2x2 table
    >>> metric.update(chi_sq, n, min_dim)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # State for storing chi-square statistics and sample info
        self.add_state("chi_square_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("min_dim_values", default=[], dist_reduce_fx="cat")

    def update(self, chi_square: Tensor, sample_size: Tensor, min_dim: Tensor) -> None:
        """
        Update the metric state with chi-square statistic and table information.

        Parameters
        ----------
        chi_square : Tensor
            Chi-square test statistic.
        sample_size : Tensor
            Total sample size (sum of all cells in contingency table).
        min_dim : Tensor
            Minimum of (number of rows - 1, number of columns - 1).
        """
        self.chi_square_values.append(chi_square)
        self.sample_size_values.append(sample_size)
        self.min_dim_values.append(min_dim)

    def compute(self) -> Tensor:
        """
        Compute Cramer's V based on stored chi-square statistics.

        Returns
        -------
        Tensor
            The computed Cramer's V effect size (between 0 and 1).
        """
        if (
            not self.chi_square_values
            or not self.sample_size_values
            or not self.min_dim_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        chi_sq = self.chi_square_values[-1]
        n = self.sample_size_values[-1]
        min_d = self.min_dim_values[-1]

        # Inline implementation of cramers_v to avoid circular imports

        # Convert inputs to tensors if needed
        chi_sq = torch.as_tensor(chi_sq)
        n = torch.as_tensor(n)
        min_d = torch.as_tensor(min_d)

        # Ensure all have the same dtype
        dtype = chi_sq.dtype
        if n.dtype != dtype:
            if n.dtype == torch.float64 or dtype == torch.float64:
                dtype = torch.float64
        if min_d.dtype != dtype:
            if min_d.dtype == torch.float64 or dtype == torch.float64:
                dtype = torch.float64

        chi_sq = chi_sq.to(dtype)
        n = n.to(dtype)
        min_d = min_d.to(dtype)

        # Cramer's V formula: V = sqrt(χ² / (n * min_dim))
        # where min_dim = min(r-1, c-1) for r rows and c columns
        output = torch.sqrt(chi_sq / (n * min_d))

        # Clamp to [0, 1] range to handle numerical errors
        return torch.clamp(output, 0.0, 1.0)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.chi_square_values = []
        self.sample_size_values = []
        self.min_dim_values = []

    def plot(
        self,
        plot_type: str = "effect_size",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Cramer's V effect size visualization.

        Parameters
        ----------
        plot_type : str, default="effect_size"
            Type of plot to create. Currently only "effect_size" is supported.
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
        >>> from beignet.metrics import CramersV
        >>> metric = CramersV()
        >>> chi_sq = torch.tensor(10.5)
        >>> n = torch.tensor(100)
        >>> min_dim = torch.tensor(1)
        >>> metric.update(chi_sq, n, min_dim)
        >>> fig = metric.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.chi_square_values
            or not self.sample_size_values
            or not self.min_dim_values
        ):
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Compute Cramer's V
        effect_size = self.compute()

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = ["Cramer's V"]
            effect_sizes = [float(effect_size)]

            bars = ax.bar(categories, effect_sizes, **kwargs)

            # Color code by effect size magnitude (Cramer's V conventions)
            color = "lightblue"
            if abs(effect_sizes[0]) >= 0.5:
                color = "red"  # Large effect
            elif abs(effect_sizes[0]) >= 0.3:
                color = "orange"  # Medium effect
            elif abs(effect_sizes[0]) >= 0.1:
                color = "yellow"  # Small effect

            bars[0].set_color(color)

            # Add reference lines for effect size thresholds
            ax.axhline(
                y=0.1,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Small (0.1)",
            )
            ax.axhline(
                y=0.3,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Medium (0.3)",
            )
            ax.axhline(
                y=0.5,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label="Large (0.5)",
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_ylabel("Cramer's V")
            ax.set_ylim(0, 1)  # Cramer's V ranges from 0 to 1
            ax.legend()

            if title is None:
                title = f"Effect Size: Cramer's V = {float(effect_size):.3f}"
        else:
            raise ValueError(f"plot_type must be 'effect_size', got {plot_type}")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
