"""Chi-square goodness-of-fit power metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._chi_squared_goodness_of_fit_power import chi_square_goodness_of_fit_power


class ChiSquareGoodnessOfFitPower(Metric):
    """Computes statistical power for chi-square goodness-of-fit tests.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. This is calculated as the square root of the
        sum of squared standardized differences: w = √(Σ((p₁ᵢ - p₀ᵢ)²/p₀ᵢ))
        where p₀ᵢ are the expected proportions and p₁ᵢ are the observed proportions.
    sample_size : Tensor
        Sample size (total number of observations).
    df : Tensor
        Degrees of freedom for the chi-square test (number of categories - 1).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> metric = ChiSquareGoodnessOfFitPower()
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(100)
    >>> df = torch.tensor(3)
    >>> metric.update(effect_size, sample_size, df)
    >>> metric.compute()
    tensor(0.6740)
    """

    full_state_update: bool = False

    def __init__(self, alpha: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("sample_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("df", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, sample_size: Tensor, df: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Cohen's w effect size.
        sample_size : Tensor
            Sample size.
        df : Tensor
            Degrees of freedom.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.sample_size = sample_size
            self.df = df
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.sample_size = (self.sample_size * self.total + sample_size) / (
                self.total + 1
            )
            self.df = (self.df * self.total + df) / (self.total + 1)
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the statistical power."""
        return chi_square_goodness_of_fit_power(
            effect_size=self.effect_size,
            sample_size=self.sample_size,
            df=self.df,
            alpha=self.alpha,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        sample_size: Optional[Any] = None,
        df: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot chi-square goodness-of-fit power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "sample_size", or "df".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        sample_size : array-like, optional
            Range of sample sizes to plot. If None, uses reasonable default range.
        df : array-like, optional
            Range of degrees of freedom to plot. If None, uses reasonable default range.
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
        >>> metric = ChiSquareGoodnessOfFitPower()
        >>> fig = metric.plot(dep_var="effect_size", df=[2, 3, 4])
        >>> fig = metric.plot(dep_var="sample_size", effect_size=[0.1, 0.3, 0.5])
        >>> fig = metric.plot(dep_var="df", effect_size=[0.3])
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
                x_values = np.linspace(0.1, 0.8, 100)
            else:
                x_values = np.asarray(effect_size)

            if df is None:
                param_values = [2, 3, 4]  # Different degrees of freedom
                param_name = "df"
                default_sample_size = 100
            else:
                param_values = np.asarray(df)
                param_name = "df"
                default_sample_size = 100

            x_label = "Effect Size (Cohen's w)"

        elif dep_var == "sample_size":
            if sample_size is None:
                x_values = np.linspace(20, 500, 100)
            else:
                x_values = np.asarray(sample_size)

            if effect_size is None:
                param_values = [0.1, 0.3, 0.5]  # Small, medium, large effects
                param_name = "Effect Size"
                default_df = 3
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_df = 3

            x_label = "Sample Size"

        elif dep_var == "df":
            if df is None:
                x_values = np.linspace(1, 10, 50)
            else:
                x_values = np.asarray(df)

            if effect_size is None:
                param_values = [0.1, 0.3, 0.5]
                param_name = "Effect Size"
                default_sample_size = 100
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_sample_size = 100

            x_label = "Degrees of Freedom"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'sample_size', or 'df', got {dep_var}"
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = chi_square_goodness_of_fit_power(
                        effect_size=torch.tensor(float(x_val)),
                        sample_size=torch.tensor(float(default_sample_size)),
                        df=torch.tensor(int(param_val)),
                        alpha=alpha,
                    )
                elif dep_var == "sample_size":
                    power_val = chi_square_goodness_of_fit_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(x_val)),
                        df=torch.tensor(int(default_df)),
                        alpha=alpha,
                    )
                elif dep_var == "df":
                    power_val = chi_square_goodness_of_fit_power(
                        effect_size=torch.tensor(float(param_val)),
                        sample_size=torch.tensor(float(default_sample_size)),
                        df=torch.tensor(int(x_val)),
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
            title = "Power Analysis: Chi-Square Goodness-of-Fit Test"
        ax.set_title(title)

        plt.tight_layout()
        return fig
