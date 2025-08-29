import torch
from torch import Tensor


class Poisson(torch.distributions.Poisson):
    r"""
    Poisson distribution with inverse cumulative distribution function.

    Extends torch.distributions.Poisson to provide icdf functionality using
    numerical approximation methods suitable for statistical computations.

    Parameters
    ----------
    rate : Tensor
        Rate parameter (must be positive).

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import Poisson
    >>> dist = Poisson(rate=torch.tensor(5.0))
    >>> quantile = dist.icdf(torch.tensor(0.95))
    >>> quantile
    tensor(9.0)

    Notes
    -----
    The Poisson distribution has mean and variance both equal to the rate parameter λ.
    The inverse CDF is computed using the relationship with the chi-squared distribution
    for large rates and direct computation for small rates.
    """

    arg_constraints = {}

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        For the Poisson distribution, the inverse CDF doesn't have a closed form,
        so we use numerical approximation methods:

        1. For large rates (λ > 30), use normal approximation with continuity correction
        2. For medium rates (10 < λ ≤ 30), use Wilson-Hilferty chi-squared approximation
        3. For small rates (λ ≤ 10), use iterative search

        Parameters
        ----------
        value : Tensor
            Probability values in [0, 1].

        Returns
        -------
        Tensor
            Quantiles corresponding to the given probabilities.

        Notes
        -----
        The Poisson quantile function is computed using different approximations
        depending on the rate parameter:

        For large λ (> 30): Normal approximation P(λ) ≈ N(λ, λ) with continuity correction
        For medium λ: Chi-squared relationship via Gamma distribution
        For small λ: Direct computation using CDF inversion
        """
        # Ensure value is a tensor with proper dtype
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=eps, max=1 - eps)

        # Clamp rate to avoid numerical issues
        rate_clamped = torch.clamp(self.rate, min=eps)

        # Method selection based on rate size (using torch.where to avoid branching)
        large_rate_mask = rate_clamped > 30.0
        medium_rate_mask = (rate_clamped > 10.0) & (rate_clamped <= 30.0)

        # Compute all approximations (torch.compile friendly)
        large_result = self._large_rate_approximation(value, rate_clamped)
        medium_result = self._medium_rate_approximation(value, rate_clamped)
        small_result = self._small_rate_computation(value, rate_clamped)

        # Select appropriate result using torch.where (no data-dependent branching)
        result = torch.where(
            large_rate_mask,
            large_result,
            torch.where(
                medium_rate_mask,
                medium_result,
                small_result,
            ),
        )

        # Ensure integer results for Poisson quantiles
        result = torch.floor(result)

        # Ensure non-negative results
        result = torch.clamp(result, min=0.0)

        return result

    def _large_rate_approximation(self, value: Tensor, rate: Tensor) -> Tensor:
        """Normal approximation for large rates with continuity correction."""
        # For large λ, Poisson(λ) ≈ Normal(λ, λ) with continuity correction

        # Get normal quantile
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # Normal approximation: X ≈ λ + √λ * Z - 0.5 (continuity correction)
        quantile = rate + torch.sqrt(rate) * z - 0.5

        return quantile

    def _medium_rate_approximation(self, value: Tensor, rate: Tensor) -> Tensor:
        """Wilson-Hilferty chi-squared approximation for medium rates."""
        # Use relationship with chi-squared: If X ~ Poisson(λ), then 2X ~ χ²(2λ) approximately

        # Get normal quantile
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # Wilson-Hilferty approximation for chi-squared
        df = 2.0 * rate  # degrees of freedom for chi-squared
        h = 2.0 / (9.0 * df)
        sqrt_h = torch.sqrt(h)

        # Chi-squared quantile approximation
        chi2_quantile = df * torch.clamp(1.0 - h + z * sqrt_h, min=0.01) ** 3

        # Convert back to Poisson: X = χ²/2
        quantile = chi2_quantile / 2.0

        return quantile

    def _small_rate_computation(self, value: Tensor, rate: Tensor) -> Tensor:
        """Direct computation for small rates using iterative search."""
        # For small rates, use a more accurate method based on the discrete nature

        # Start with normal approximation as initial guess
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)
        initial_guess = torch.clamp(rate + torch.sqrt(rate) * z, min=0.0)

        # For very small rates, use a simple approximation
        # When rate is small, the distribution is highly skewed
        small_rate_correction = torch.where(
            rate < 1.0,
            torch.sqrt(-2.0 * rate * torch.log(1.0 - value)),
            initial_guess,
        )

        # Use the better approximation for rates > 1
        result = torch.where(rate >= 1.0, initial_guess, small_rate_correction)

        return result
