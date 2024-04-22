from typing import Dict


def _default_nhc_kwargs(tau: float, overrides: Dict) -> Dict:
    # Copied from https://github.com/jax-md/jax-md/blob/e0ea7d3c235724aac60cecad8dc6f4571d7e5757/jax_md/simulate.py#L485
    default_kwargs = {"size": 3, "steps": 2, "system_steps": 3, "oscillation": tau}

    if overrides is None:
        return default_kwargs

    return {key: overrides.get(key, default_kwargs[key]) for key in default_kwargs}
