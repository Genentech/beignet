import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("beignet")
except PackageNotFoundError:
    __version__ = None

from ._bisect import bisect
from ._chandrupatla import chandrupatla
from ._convolve import convolve
from ._default_dtype_manager import default_dtype_manager
from ._dihedral_angle import dihedral_angle
from ._farthest_first_traversal import farthest_first_traversal
from ._identity_matrix import identity_matrix
from ._kabsch import kabsch
from ._lennard_jones_potential import lennard_jones_potential
from ._optional_dependencies import optional_dependencies
from ._pad import pad_to_target_length
from ._radius import radius, radius_graph
from ._root_scalar import root_scalar

__all__ = [
    "bisect",
    "chandrupatla",
    "convolve",
    "default_dtype_manager",
    "dihedral_angle",
    "farthest_first_traversal",
    "identity_matrix",
    "kabsch",
    "lennard_jones_potential",
    "optional_dependencies",
    "pad_to_target_length",
    "radius",
    "radius_graph",
    "root_scalar",
]
