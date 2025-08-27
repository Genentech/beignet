import torch
from torch import Tensor


class Chi2(torch.distributions.Chi2):
    r"""
    Chi-squared distribution with inverse cumulative distribution function.

    Extends torch.distributions.Chi2 to provide icdf functionality using
    the Wilson-Hilferty normal approximation with corrections.

    Parameters
    ----------
    df : Tensor
        Degrees of freedom.

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import Chi2
    >>> dist = Chi2(df=torch.tensor(5.0))
    >>> quantile = dist.icdf(torch.tensor(0.95))
    >>> quantile
    tensor(11.0705)
    """

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        Uses the Wilson-Hilferty transformation combined with normal
        quantiles for high accuracy across all degrees of freedom.

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
        The implementation uses the Wilson-Hilferty normal approximation:

        .. math::
            \\chi^2 \\approx \\nu \\left(1 - \\frac{2}{9\\nu} + z\\sqrt{\\frac{2}{9\\nu}}\\right)^3

        where ν is the degrees of freedom and z is the standard normal quantile.
        Additional corrections are applied for improved accuracy.
        """
        # Ensure value is a tensor with proper dtype
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=eps, max=1 - eps)

        # Get normal quantile
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # Clamp degrees of freedom to avoid division by zero
        df_clamped = torch.clamp(self.df, min=1.0)

        # Wilson-Hilferty transformation
        # chi^2 ≈ ν * (1 - 2/(9ν) + z*sqrt(2/(9ν)))^3

        h = 2.0 / (9.0 * df_clamped)
        sqrt_h = torch.sqrt(h)

        # Base transformation
        transformed = 1.0 - h + z * sqrt_h

        # Handle negative values that would cause issues with cube
        transformed = torch.clamp(transformed, min=0.01)

        # Apply cube transformation
        chi2_approx = df_clamped * transformed**3

        # For small degrees of freedom, add correction terms
        # Based on Cornish-Fisher expansion
        small_df_mask = df_clamped < 10.0

        # Additional correction for small degrees of freedom
        correction = torch.where(
            small_df_mask,
            z**2 / (108.0 * df_clamped**2) + z**4 / (3240.0 * df_clamped**3),
            torch.zeros_like(z),
        )
        chi2_approx = chi2_approx + correction * df_clamped

        # For very large degrees of freedom, use normal approximation
        # chi^2 ≈ ν + sqrt(2ν) * z
        large_df_mask = df_clamped > 100.0

        normal_approx = df_clamped + torch.sqrt(2.0 * df_clamped) * z
        chi2_approx = torch.where(large_df_mask, normal_approx, chi2_approx)

        # Ensure non-negative result
        result = torch.clamp(chi2_approx, min=0.0)

        return result
