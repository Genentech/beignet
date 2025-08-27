import torch
from torch import Tensor


class NonCentralT(torch.distributions.Distribution):
    r"""
    Non-central t-distribution with inverse cumulative distribution function.

    The non-central t-distribution arises in statistics when testing hypotheses
    about means with non-zero effect sizes. It's fundamental to power analysis
    for t-tests, confidence intervals, and regression analysis.

    Parameters
    ----------
    df : Tensor
        Degrees of freedom (must be positive).
    nc : Tensor
        Non-centrality parameter (can be any real value).

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import NonCentralT
    >>> dist = NonCentralT(torch.tensor(10.0), torch.tensor(2.0))
    >>> quantile = dist.icdf(torch.tensor(0.975))
    >>> quantile
    tensor(4.3)

    Notes
    -----
    The non-central t-distribution with degrees of freedom ν and non-centrality
    parameter δ has mean approximately δ√(π/2) Γ((ν-1)/2) / Γ(ν/2) for ν > 1,
    and variance approximately ν(1 + δ²)/(ν-2) - (mean)² for ν > 2.

    The implementation uses Cornish-Fisher corrections and normal approximations
    optimized for different parameter ranges.
    """

    arg_constraints = {}

    def __init__(self, df: Tensor, nc: Tensor, validate_args: bool | None = None):
        """
        Initialize NonCentralT distribution.

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

        Uses a combination of methods optimized for different parameter ranges:
        - Central t-distribution for nc ≈ 0
        - Normal approximation for large degrees of freedom
        - Cornish-Fisher corrections for moderate parameters

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
        The implementation adapts between several methods:

        1. **Central case (nc ≈ 0)**: Uses standard t-distribution
        2. **Large df**: Normal approximation with non-centrality shift
        3. **Moderate parameters**: Cornish-Fisher expansion with corrections
        """
        # Ensure value is a tensor with proper dtype and device
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=eps, max=1 - eps)

        # Clamp parameters for numerical stability
        df_clamped = torch.clamp(self.df, min=1.0)  # Need df >= 1 for t-distribution
        nc_clamped = self.nc  # Non-centrality can be any real number

        # Get standard normal quantile
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # Method selection based on parameters
        central_mask = torch.abs(nc_clamped) < eps
        large_df_mask = df_clamped > 100.0

        # Method 1: Central t-distribution (nc ≈ 0)
        # Use Cornish-Fisher correction for central t
        correction = z / (4.0 * df_clamped) * (z**2 + 1.0)
        correction2 = z / (96.0 * df_clamped**2) * (5.0 * z**4 + 16.0 * z**2 + 3.0)
        correction3 = (
            z
            / (384.0 * df_clamped**3)
            * (3.0 * z**6 + 19.0 * z**4 + 17.0 * z**2 - 15.0)
        )
        t_central = z + correction + correction2 + correction3

        # Method 2: Large degrees of freedom (approximately normal)
        # For large df, non-central t approaches normal with location shift
        t_large_df = z + nc_clamped

        # Method 3: Moderate parameters with non-centrality
        # Use approximation: non-central t ≈ central t + non-centrality adjustment

        # Variance inflation due to non-centrality (with numerical stability)
        variance_inflation = torch.sqrt(
            1.0 + nc_clamped**2 / torch.clamp(df_clamped, min=2.0),
        )

        # Approximate non-central t quantile
        t_approx = nc_clamped + variance_inflation * t_central

        # For moderate non-centrality, blend between methods
        moderate_nc_mask = (
            (torch.abs(nc_clamped) <= 5.0) & (~central_mask) & (~large_df_mask)
        )

        # Improved approximation for moderate non-centrality
        # Uses a shifted and scaled t-distribution
        shift = nc_clamped
        df_safe = torch.clamp(
            df_clamped,
            min=3.0,
        )  # Ensure df > 2 for variance to exist
        scale_factor = torch.sqrt(
            df_safe / torch.clamp(df_safe - 2.0 + nc_clamped**2, min=1.0),
        )
        scale_factor = torch.clamp(
            scale_factor,
            min=0.5,
            max=2.0,
        )  # Prevent extreme scaling

        t_moderate = shift + scale_factor * t_central

        # Select appropriate method
        result = torch.where(
            central_mask,
            t_central,
            torch.where(
                large_df_mask,
                t_large_df,
                torch.where(
                    moderate_nc_mask,
                    t_moderate,
                    t_approx,
                ),
            ),
        )

        return result

    def cdf(self, value: Tensor) -> Tensor:
        r"""
        Cumulative distribution function.

        Computes P(X ≤ value) where X follows a non-central t-distribution.
        Uses different approximations based on parameter ranges for optimal accuracy.

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
        The implementation uses multiple strategies:

        1. **Central case (nc ≈ 0)**: Standard t-distribution CDF
        2. **Large df**: Normal approximation with location shift
        3. **Small to moderate df**: Normal approximation with mean/variance matching
        4. **Large |nc|**: Normal approximation with improved moments
        """
        # Ensure value is a tensor with proper dtype and device
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Clamp parameters for numerical stability
        df_clamped = torch.clamp(self.df, min=1.0)
        nc_clamped = self.nc
        eps = torch.finfo(value.dtype).eps

        # Method selection based on parameters
        central_mask = torch.abs(nc_clamped) < eps
        large_df_mask = df_clamped > 30.0

        # Method 1: Central t-distribution (nc ≈ 0)
        # Use the standard t-distribution CDF
        # P(T_ν ≤ t) = 0.5 + (t/√(ν)) * F(0.5, (ν+1)/2; 1.5; -t²/ν) where F is hypergeometric
        # For numerical stability, use the relationship with incomplete beta function

        # Standard t-distribution CDF using beta function relationship
        # P(T ≤ t) = 0.5 + 0.5 * sign(t) * I_x(1/2, ν/2) where x = t²/(t²+ν)
        t_squared = value**2
        x_beta = t_squared / (t_squared + df_clamped)
        sign_t = torch.sign(value)

        # Handle the beta function computation
        half_one = torch.tensor(0.5, dtype=value.dtype, device=value.device)
        half_df = df_clamped / 2.0

        # Use regularized incomplete beta function
        beta_cdf = torch.special.betainc(half_one, half_df, x_beta)
        central_cdf = 0.5 + 0.5 * sign_t * beta_cdf
        central_cdf = torch.clamp(central_cdf, min=0.0, max=1.0)

        # Method 2: Large degrees of freedom (approximately normal)
        # For large df, non-central t ≈ Normal(nc, 1)
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z_large_df = value - nc_clamped
        large_df_cdf = 0.5 * (1.0 + torch.erf(z_large_df / sqrt_two))

        # Method 3: Normal approximation using distribution moments
        # Use the approximate mean and variance for the normal approximation
        mean_nct = self.mean
        variance_nct = self.variance
        std_nct = torch.sqrt(torch.clamp(variance_nct, min=eps))

        z_normal = (value - mean_nct) / torch.clamp(std_nct, min=eps)
        normal_cdf = 0.5 * (1.0 + torch.erf(z_normal / sqrt_two))

        # Method 4: Improved approximation for large non-centrality
        # When |nc| is large, use a shifted normal approximation
        large_nc_mask = torch.abs(nc_clamped) > 5.0

        # For large nc, the distribution becomes approximately normal
        # with mean ≈ nc and variance ≈ 1 + nc²/df (simplified)
        mean_large_nc = nc_clamped
        var_large_nc = 1.0 + nc_clamped**2 / torch.clamp(df_clamped, min=2.0)
        std_large_nc = torch.sqrt(var_large_nc)

        z_large_nc = (value - mean_large_nc) / torch.clamp(std_large_nc, min=eps)
        large_nc_cdf = 0.5 * (1.0 + torch.erf(z_large_nc / sqrt_two))

        # Combine methods based on parameter regions
        result = torch.where(
            central_mask,
            central_cdf,
            torch.where(
                large_df_mask,
                large_df_cdf,
                torch.where(
                    large_nc_mask,
                    large_nc_cdf,
                    normal_cdf,  # Default to moment-matched normal approximation
                ),
            ),
        )

        return torch.clamp(result, min=0.0, max=1.0)

    @property
    def mean(self) -> Tensor:
        """
        Approximate mean of the distribution.

        For df > 1, the mean is approximately nc * sqrt(π/2) * Γ((df-1)/2) / Γ(df/2).
        For large df, this approaches nc.
        """
        # Simplified approximation that works well for df > 2
        df_clamped = torch.clamp(self.df, min=2.0)
        correction_factor = torch.sqrt(df_clamped / 2.0) * torch.exp(
            torch.lgamma((df_clamped - 1.0) / 2.0) - torch.lgamma(df_clamped / 2.0),
        )
        return self.nc * correction_factor

    @property
    def variance(self) -> Tensor:
        """
        Approximate variance of the distribution.

        For df > 2, approximately df(1 + nc²)/(df-2) - (mean)².
        """
        df_clamped = torch.clamp(self.df, min=2.1)  # Need df > 2 for variance
        mean_val = self.mean
        variance_approx = (
            df_clamped * (1.0 + self.nc**2) / (df_clamped - 2.0) - mean_val**2
        )
        return torch.clamp(variance_approx, min=1.0)  # Ensure positive variance
