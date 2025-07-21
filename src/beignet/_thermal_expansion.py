import torch
from torch import Tensor


def thermal_expansion(
    input: Tensor,
    temperature: Tensor,
    energies: Tensor,
) -> Tensor:
    r"""
    Calculate thermal expansion coefficient from volume and enthalpy fluctuations.

    The isobaric thermal expansion coefficient α is calculated using the
    fluctuation formula from the NPT ensemble:

    α = (⟨HV⟩ - ⟨H⟩⟨V⟩) / (kB T² ⟨V⟩)

    where ⟨HV⟩ is the ensemble average of the product of enthalpy and volume,
    ⟨H⟩ and ⟨V⟩ are the ensemble averages of enthalpy and volume respectively,
    kB is the Boltzmann constant, and T is the absolute temperature.

    Parameters
    ----------
    input : Tensor, shape=(..., N)
        Volume time series data from NPT molecular dynamics simulations.
        The last dimension contains volume measurements at different timesteps.
        Units should be in Å³ (cubic Angstroms).
    temperature : Tensor, shape=(...) or scalar
        Absolute temperature in Kelvin. Must be broadcastable with input.shape[:-1].
    energies : Tensor, shape=(..., N)
        Enthalpy time series data (H = E + PV) from NPT simulations.
        Must have the same shape as input. Units should be in eV.

    Returns
    -------
    alpha : Tensor, shape=(...)
        Thermal expansion coefficient in units of K⁻¹ (per Kelvin).

    Notes
    -----
    The Boltzmann constant is used in units of eV/K to match
    the atomic units convention (Å³ for volume, eV for energy).

    kB = 8.617333262145e-5 eV/K

    This formula is derived from statistical mechanics and provides
    a direct route to calculate thermal properties from molecular
    dynamics trajectories in the NPT ensemble.

    Examples
    --------
    >>> # Single trajectory
    >>> volumes = torch.randn(1000) * 10 + 5000  # 1000 timesteps
    >>> enthalpies = torch.randn(1000) * 100 - 5000  # Enthalpy values
    >>> temp = torch.tensor(300.0)  # 300 K
    >>> alpha = beignet.thermal_expansion(volumes, temp, enthalpies)
    >>> alpha.shape
    torch.Size([])

    >>> # Batch of trajectories
    >>> volumes = torch.randn(5, 1000) * 10 + 5000  # 5 trajectories
    >>> enthalpies = torch.randn(5, 1000) * 100 - 5000
    >>> temps = torch.tensor([250, 300, 350, 400, 450])  # Different temperatures
    >>> alpha = beignet.thermal_expansion(volumes, temps, enthalpies)
    >>> alpha.shape
    torch.Size([5])
    """
    # Boltzmann constant in eV/K for atomic units
    kB = 8.617333262145e-5

    # Ensure temperature is a tensor
    if not isinstance(temperature, Tensor):
        temperature = torch.tensor(temperature, dtype=input.dtype, device=input.device)

    # Ensure all tensors have the same dtype and device
    energies = energies.to(dtype=input.dtype, device=input.device)
    temperature = temperature.to(dtype=input.dtype, device=input.device)

    # Check shapes match
    if input.shape != energies.shape:
        raise ValueError(
            f"input and energies must have the same shape, "
            f"got {input.shape} and {energies.shape}"
        )

    # Calculate ensemble averages along the last dimension (timesteps)
    mean_volume = input.mean(dim=-1)
    mean_enthalpy = energies.mean(dim=-1)

    # Calculate covariance using the numerically stable formula
    # Cov(H,V) = E[(H - E[H])(V - E[V])]
    volume_centered = input - mean_volume.unsqueeze(-1)
    enthalpy_centered = energies - mean_enthalpy.unsqueeze(-1)
    covariance = (volume_centered * enthalpy_centered).mean(dim=-1)

    # Calculate thermal expansion coefficient
    # α = covariance / (kB * T² * ⟨V⟩)
    alpha = covariance / (kB * temperature * temperature * mean_volume)

    return alpha
