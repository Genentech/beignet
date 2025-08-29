from typing import Any, Optional

from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import partial_eta_squared


class PartialEtaSquared(Metric):
    r"""
    Compute Partial Eta Squared effect size metric.

    This metric calculates Partial Eta Squared, which measures the proportion
    of variance in the dependent variable that is associated with a specific
    factor while controlling for other factors. It follows TorchMetrics
    conventions for stateful metric computation.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import PartialEtaSquared
    >>> metric = PartialEtaSquared()
    >>> sum_of_squares_effect = torch.tensor(25.0)
    >>> sum_of_squares_error = torch.tensor(75.0)
    >>> metric.update(sum_of_squares_effect, sum_of_squares_error)
    >>> metric.compute()
    tensor(0.2500)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # State for storing sum of squares values
        self.add_state("ss_effect_values", default=[], dist_reduce_fx="cat")
        self.add_state("ss_error_values", default=[], dist_reduce_fx="cat")

    def update(
        self, sum_of_squares_effect: Tensor, sum_of_squares_error: Tensor
    ) -> None:
        """
        Update the metric state with sum of squares values.

        Parameters
        ----------
        sum_of_squares_effect : Tensor
            Sum of squares for the specific effect/factor.
        sum_of_squares_error : Tensor
            Sum of squares for error/residual.
        """
        self.ss_effect_values.append(sum_of_squares_effect)
        self.ss_error_values.append(sum_of_squares_error)

    def compute(self) -> Tensor:
        """
        Compute Partial Eta Squared based on stored sum of squares values.

        Returns
        -------
        Tensor
            The computed Partial Eta Squared effect size.
        """
        if not self.ss_effect_values or not self.ss_error_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        ss_effect = self.ss_effect_values[-1]
        ss_error = self.ss_error_values[-1]

        # Use functional implementation
        return partial_eta_squared(ss_effect, ss_error)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.ss_effect_values = []
        self.ss_error_values = []

    def plot(
        self,
        plot_type: str = "effect_size",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Partial Eta Squared effect size visualization.

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
        >>> from beignet.metrics import PartialEtaSquared
        >>> metric = PartialEtaSquared()
        >>> ss_effect = torch.tensor(25.0)
        >>> ss_error = torch.tensor(75.0)
        >>> metric.update(ss_effect, ss_error)
        >>> fig = metric.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.ss_effect_values or not self.ss_error_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Compute Partial Eta Squared
        effect_size = self.compute()

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_size":
            # Create a simple effect size visualization
            categories = ["Partial Eta Squared"]
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

            ax.set_ylabel("Partial Eta Squared")
            ax.set_ylim(0, 1)  # Partial eta squared ranges from 0 to 1
            ax.legend()

            if title is None:
                title = f"Effect Size: Partial Eta Squared = {float(effect_size):.3f}"
        else:
            raise ValueError(f"plot_type must be 'effect_size', got {plot_type}")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
