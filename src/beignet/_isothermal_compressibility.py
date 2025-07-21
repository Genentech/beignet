import torch
from torch import Tensor


def isothermal_compressibility(
    input: Tensor,
    temperature: Tensor,
) -> Tensor:
    r"""
    Calculate isothermal compressibility from volume fluctuations.

    The isothermal compressibility κT is calculated using the fluctuation formula
    from statistical mechanics:

    κT = ⟨ΔV²⟩ / (kB T ⟨V⟩)

    where ⟨ΔV²⟩ is the variance of volume, kB is Boltzmann constant,
    T is temperature, and ⟨V⟩ is the mean volume.

    Parameters
    ----------
    input : Tensor, shape=(..., N)
        Volume time series data from molecular dynamics simulations.
        The last dimension contains volume measurements at different timesteps.
        Units should be in Å³ (cubic Angstroms).
    temperature : Tensor, shape=(...) or scalar
        Temperature in Kelvin. Must be broadcastable with input.shape[:-1].

    Returns
    -------
    compressibility : Tensor, shape=(...)
        Isothermal compressibility in units of eV⁻¹ Å³
        (inverse energy per volume in atomic units).

    Notes
    -----
    The Boltzmann constant is used in units of eV/K to match
    the atomic units convention (Å³ for volume, eV for energy).

    kB = 8.617333262145e-5 eV/K

    Examples
    --------
    >>> # Single trajectory
    >>> volumes = torch.randn(1000) * 10 + 1000  # 1000 timesteps
    >>> temp = torch.tensor(300.0)  # 300 K
    >>> kappa = beignet.isothermal_compressibility(volumes, temp)
    >>> kappa.shape
    torch.Size([])

    >>> # Batch of trajectories
    >>> volumes = torch.randn(5, 1000) * 10 + 1000  # 5 trajectories
    >>> temps = torch.tensor([250, 300, 350, 400, 450])  # Different temperatures
    >>> kappa = beignet.isothermal_compressibility(volumes, temps)
    >>> kappa.shape
    torch.Size([5])
    """
    # Boltzmann constant in eV/K for atomic units
    kB = 8.617333262145e-5

    # Ensure temperature is a tensor
    if not isinstance(temperature, Tensor):
        temperature = torch.tensor(temperature, dtype=input.dtype, device=input.device)

    # Calculate mean volume along the last dimension (timesteps)
    mean_volume = input.mean(dim=-1)

    # Calculate variance of volume
    # Use PyTorch's built-in variance function for numerical stability
    variance = input.var(dim=-1, unbiased=False)

    # Ensure variance is non-negative (handle numerical precision issues)
    variance = torch.clamp(variance, min=0.0)

    # Calculate isothermal compressibility
    # κT = ⟨ΔV²⟩ / (kB * T * ⟨V⟩)
    compressibility = variance / (kB * temperature * mean_volume)

    return compressibility
