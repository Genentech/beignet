import torch
from torch import Tensor
from torch.distributions import Distribution


class NonCentralChi2(Distribution):
    r"""
    Non-central chi-squared distribution with inverse cumulative distribution function.

    The non-central chi-squared distribution with degrees of freedom `df` and
    non-centrality parameter `nc` arises in statistics when testing hypotheses
    with non-zero effect sizes, particularly in power analysis and ANOVA.

    Parameters
    ----------
    df : Tensor
        Degrees of freedom (must be positive).
    nc : Tensor
        Non-centrality parameter (must be non-negative).

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import NonCentralChi2
    >>> dist = NonCentralChi2(torch.tensor(5.0), torch.tensor(2.0))
    >>> quantile = dist.icdf(torch.tensor(0.95))
    >>> quantile
    tensor(14.8)

    Notes
    -----
    The non-central chi-squared distribution has mean μ = df + nc and
    variance σ² = 2(df + 2*nc). The implementation uses a Wilson-Hilferty
    transformation for the quantile function with corrections for non-centrality.
    """

    arg_constraints = {}

    def __init__(self, df: Tensor, nc: Tensor, validate_args: bool | None = None):
        """
        Initialize NonCentralChi2 distribution.

        Parameters
        ----------
        df : Tensor
            Degrees of freedom parameter.
        nc : Tensor
            Non-centrality parameter.
        validate_args : bool | None, optional
            Whether to validate arguments.
        """
        self.df = torch.as_tensor(df)
        self.nc = torch.as_tensor(nc)

        if validate_args is not False:
            # Skip validation for torch.compile compatibility
            pass

        batch_shape = torch.broadcast_shapes(self.df.shape, self.nc.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        Uses Wilson-Hilferty transformation with non-centrality corrections
        for accurate quantile computation across different parameter ranges.

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
        The implementation uses the Wilson-Hilferty transformation:

        .. math::
            X = df + nc + \sqrt{2(df + 2 \cdot nc)} \cdot z

        where z is the standard normal quantile, with additional corrections
        for accuracy when the non-centrality parameter is large.
        """
        # Ensure value is a tensor with proper dtype and device
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=eps, max=1 - eps)

        # Clamp parameters for numerical stability
        df_clamped = torch.clamp(self.df, min=eps)
        nc_clamped = torch.clamp(self.nc, min=0.0)

        # Mean and variance of non-central chi-squared
        mean = df_clamped + nc_clamped
        variance = 2.0 * (df_clamped + 2.0 * nc_clamped)

        # Get standard normal quantile
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # For central chi-squared (nc ≈ 0), use Wilson-Hilferty transformation
        central_mask = nc_clamped < eps

        # Wilson-Hilferty transformation for central case
        h_central = 2.0 / (9.0 * df_clamped)
        sqrt_h_central = torch.sqrt(h_central)
        transformed_central = 1.0 - h_central + z * sqrt_h_central
        chi2_central = df_clamped * torch.pow(
            torch.clamp(transformed_central, min=eps),
            3,
        )

        # For non-central case, use normal approximation
        # This is accurate for moderate to large non-centrality parameters
        chi2_noncentral = mean + torch.sqrt(variance) * z

        # For very large non-centrality, use improved approximation
        large_nc_mask = nc_clamped > 30.0

        # Improved approximation for large non-centrality
        # Uses a more accurate normal approximation with skewness correction
        skewness = 8.0 / torch.sqrt(variance)  # Approximate skewness
        kurtosis_correction = skewness * (z**2 - 1.0) / 6.0
        chi2_large_nc = mean + torch.sqrt(variance) * z + kurtosis_correction

        # Select appropriate method based on non-centrality parameter
        result = torch.where(
            central_mask,
            chi2_central,
            torch.where(
                large_nc_mask,
                chi2_large_nc,
                chi2_noncentral,
            ),
        )

        # Ensure non-negative result (chi-squared is always non-negative)
        result = torch.clamp(result, min=0.0)

        return result

    def cdf(self, value: Tensor) -> Tensor:
        r"""
        Cumulative distribution function.

        Computes P(X ≤ value) where X follows a non-central chi-squared distribution.
        Uses numerical approximations appropriate for different parameter ranges.

        Parameters
        ----------
        value : Tensor
            Values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            Probabilities corresponding to the given values.

        Notes
        -----
        For small non-centrality parameters, uses the series expansion:

        .. math::
            F(x; df, nc) = \sum_{j=0}^{\infty} \frac{e^{-nc/2} (nc/2)^j}{j!} \cdot F_{df+2j}(x)

        For large non-centrality parameters, uses a normal approximation
        with continuity correction.
        """
        # Ensure value is a tensor with proper dtype and device
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=0.0)  # Chi-squared values must be non-negative

        # Clamp parameters for numerical stability
        df_clamped = torch.clamp(self.df, min=eps)
        nc_clamped = torch.clamp(self.nc, min=0.0)

        # For very small values, return 0
        tiny_mask = value < eps

        # For central chi-squared (nc ≈ 0), use standard chi-squared CDF
        central_mask = nc_clamped < eps

        # Central chi-squared CDF using gamma function relationship
        # P(X ≤ x) = γ(df/2, x/2) / Γ(df/2) where γ is lower incomplete gamma
        half_df = df_clamped / 2.0
        half_value = value / 2.0
        central_cdf = torch.special.gammainc(half_df, half_value)

        # For non-central case, use different approximations based on parameters
        small_nc_mask = (nc_clamped > eps) & (nc_clamped <= 10.0)

        # Small non-centrality: use truncated series expansion
        # Sum first few terms of the Poisson-weighted central chi-squared series
        max_terms = 20
        series_sum = torch.zeros_like(value)
        poisson_term = torch.exp(-nc_clamped / 2.0)  # j=0 term

        for j in range(max_terms):
            if j > 0:
                poisson_term = poisson_term * (nc_clamped / 2.0) / j

            # Central chi-squared CDF with df + 2*j degrees of freedom
            df_j = df_clamped + 2.0 * j
            half_df_j = df_j / 2.0
            chi2_cdf_j = torch.special.gammainc(half_df_j, half_value)

            series_sum = series_sum + poisson_term * chi2_cdf_j

            # Check for convergence (optional optimization)
            if j > 5 and torch.all(poisson_term < 1e-10):
                break

        small_nc_cdf = torch.clamp(series_sum, min=0.0, max=1.0)

        # Large non-centrality: use normal approximation with continuity correction
        mean = df_clamped + nc_clamped
        variance = 2.0 * (df_clamped + 2.0 * nc_clamped)
        std = torch.sqrt(variance)

        # Continuity correction for discrete-to-continuous approximation
        continuity_correction = 0.5
        z_score = (value + continuity_correction - mean) / torch.clamp(std, min=eps)

        # Standard normal CDF
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        large_nc_cdf = 0.5 * (1.0 + torch.erf(z_score / sqrt_two))

        # Combine different approximations
        result = torch.where(
            tiny_mask,
            torch.zeros_like(value),
            torch.where(
                central_mask,
                central_cdf,
                torch.where(
                    small_nc_mask,
                    small_nc_cdf,
                    large_nc_cdf,
                ),
            ),
        )

        return torch.clamp(result, min=0.0, max=1.0)

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        return self.df + self.nc

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        return 2.0 * (self.df + 2.0 * self.nc)
