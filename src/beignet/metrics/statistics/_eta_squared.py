from typing import Any, Optional

from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import eta_squared


class EtaSquared(Metric):
    r"""
    Compute Eta Squared effect size metric.

    This metric calculates Eta Squared, which measures the proportion of
    variance in the dependent variable that is associated with the independent
    variable(s). It follows TorchMetrics conventions for stateful metric
    computation.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import EtaSquared
    >>> metric = EtaSquared()
    >>> sum_of_squares_between = torch.tensor(25.0)
    >>> sum_of_squares_total = torch.tensor(100.0)
    >>> metric.update(sum_of_squares_between, sum_of_squares_total)
    >>> metric.compute()
    tensor(0.2500)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # State for storing sum of squares values
        self.add_state("ss_between_values", default=[], dist_reduce_fx="cat")
        self.add_state("ss_total_values", default=[], dist_reduce_fx="cat")

    def update(
        self, sum_of_squares_between: Tensor, sum_of_squares_total: Tensor
    ) -> None:
        """
        Update the metric state with sum of squares values.

        Parameters
        ----------
        sum_of_squares_between : Tensor
            Sum of squares between groups.
        sum_of_squares_total : Tensor
            Total sum of squares.
        """
        self.ss_between_values.append(sum_of_squares_between)
        self.ss_total_values.append(sum_of_squares_total)

    def compute(self) -> Tensor:
        """
        Compute Eta Squared based on stored sum of squares values.

        Returns
        -------
        Tensor
            The computed Eta Squared effect size.
        """
        if not self.ss_between_values or not self.ss_total_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        ss_between = self.ss_between_values[-1]
        ss_total = self.ss_total_values[-1]

        # Use functional implementation
        return eta_squared(ss_between, ss_total)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.ss_between_values = []
        self.ss_total_values = []

    def plot(
        self,
        plot_type: str = "effect_size",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Eta Squared effect size visualization.

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
        >>> from beignet.metrics import EtaSquared
        >>> metric = EtaSquared()
        >>> ss_between = torch.tensor(25.0)
        >>> ss_total = torch.tensor(100.0)
        >>> metric.update(ss_between, ss_total)
        >>> fig = metric.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.ss_between_values or not self.ss_total_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Compute Eta Squared
        effect_size = self.compute()

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = ["Eta Squared"]
            effect_sizes = [float(effect_size)]

            bars = ax.bar(categories, effect_sizes, **kwargs)

            # Color code by effect size magnitude (Cohen's conventions for eta squared)
            color = "lightblue"
            if abs(effect_sizes[0]) >= 0.14:
                color = "red"  # Large effect
            elif abs(effect_sizes[0]) >= 0.06:
                color = "orange"  # Medium effect
            elif abs(effect_sizes[0]) >= 0.01:
                color = "yellow"  # Small effect

            bars[0].set_color(color)

            # Add reference lines for effect size thresholds
            ax.axhline(
                y=0.01,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Small (0.01)",
            )
            ax.axhline(
                y=0.06,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Medium (0.06)",
            )
            ax.axhline(
                y=0.14,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label="Large (0.14)",
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_ylabel("Eta Squared")
            ax.set_ylim(0, 1)  # Eta squared ranges from 0 to 1
            ax.legend()

            if title is None:
                title = f"Effect Size: Eta Squared = {float(effect_size):.3f}"
        else:
            raise ValueError(f"plot_type must be 'effect_size', got {plot_type}")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
