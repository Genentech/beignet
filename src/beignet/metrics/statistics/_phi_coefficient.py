from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class PhiCoefficient(Metric):
    r"""
    Compute phi coefficient effect size for 2×2 chi-square tests.

    This metric calculates the phi coefficient, a measure of association
    for two binary variables based on chi-square statistics. It follows
    TorchMetrics conventions for stateful metric computation.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import PhiCoefficient
    >>> metric = PhiCoefficient()
    >>> chi_sq = torch.tensor(6.25)
    >>> n = torch.tensor(100)
    >>> metric.update(chi_sq, n)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # State for storing chi-square statistics and sample sizes
        self.add_state("chi_square_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")

    def update(self, chi_square: Tensor, sample_size: Tensor) -> None:
        """
        Update the metric state with chi-square statistic and sample size.

        Parameters
        ----------
        chi_square : Tensor
            Chi-square test statistic from a 2×2 contingency table.
        sample_size : Tensor
            Total sample size (sum of all cells in contingency table).
        """
        self.chi_square_values.append(chi_square)
        self.sample_size_values.append(sample_size)

    def compute(self) -> Tensor:
        """
        Compute phi coefficient based on stored chi-square statistics.

        Returns
        -------
        Tensor
            The computed phi coefficient (between 0 and 1, absolute value).
        """
        if not self.chi_square_values or not self.sample_size_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        chi_sq = self.chi_square_values[-1]
        n = self.sample_size_values[-1]

        # Inline implementation of phi_coefficient to avoid circular imports

        # Convert inputs to tensors if needed
        chi_sq = torch.as_tensor(chi_sq)
        n = torch.as_tensor(n)

        # Ensure both have the same dtype
        if chi_sq.dtype != n.dtype:
            if chi_sq.dtype == torch.float64 or n.dtype == torch.float64:
                chi_sq = chi_sq.to(torch.float64)
                n = n.to(torch.float64)
            else:
                chi_sq = chi_sq.to(torch.float32)
                n = n.to(torch.float32)

        # Phi coefficient formula: φ = sqrt(χ² / n)
        # Note: This gives the absolute value of phi. The sign would need
        # to be determined from the original contingency table structure.
        output = torch.sqrt(chi_sq / n)

        # Clamp to [0, 1] range (absolute value)
        return torch.clamp(output, 0.0, 1.0)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.chi_square_values = []
        self.sample_size_values = []

    def plot(
        self,
        plot_type: str = "effect_size",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot phi coefficient effect size visualization.

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
        >>> from beignet.metrics import PhiCoefficient
        >>> metric = PhiCoefficient()
        >>> chi_sq = torch.tensor(6.25)
        >>> n = torch.tensor(100)
        >>> metric.update(chi_sq, n)
        >>> fig = metric.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.chi_square_values or not self.sample_size_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Compute phi coefficient
        effect_size = self.compute()

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = ["Phi Coefficient"]
            effect_sizes = [float(effect_size)]

            bars = ax.bar(categories, effect_sizes, **kwargs)

            # Color code by effect size magnitude (phi coefficient conventions)
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

            ax.set_ylabel("Phi Coefficient")
            ax.set_ylim(0, 1)  # Phi coefficient ranges from 0 to 1
            ax.legend()

            if title is None:
                title = f"Effect Size: Phi Coefficient = {float(effect_size):.3f}"
        else:
            raise ValueError(f"plot_type must be 'effect_size', got {plot_type}")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
