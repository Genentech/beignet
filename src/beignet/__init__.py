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
from ._one_hot import one_hot
from ._optional_dependencies import optional_dependencies
from ._pad import pad_to_target_length
from ._radius import radius, radius_graph
from ._root_scalar import root_scalar
from .nn.functional.alphafold3._compute_alignment_error import compute_alignment_error
from .nn.functional.alphafold3._distogram_loss import distogram_loss
from .nn.functional.alphafold3._express_coordinates_in_frame import (
    express_coordinates_in_frame,
)
from .nn.functional.alphafold3._frame_aligned_point_error import (
    frame_aligned_point_error,
)
from .nn.functional.alphafold3._smooth_local_distance_difference_test import (
    smooth_local_distance_difference_test,
)
from .nn.functional.alphafold3._weighted_rigid_align import weighted_rigid_align

__all__ = [
    "bisect",
    "chandrupatla",
    "compute_alignment_error",
    "convolve",
    "default_dtype_manager",
    "dihedral_angle",
    "distogram_loss",
    "express_coordinates_in_frame",
    "farthest_first_traversal",
    "frame_aligned_point_error",
    "identity_matrix",
    "kabsch",
    "lennard_jones_potential",
    "one_hot",
    "optional_dependencies",
    "pad_to_target_length",
    "radius",
    "radius_graph",
    "root_scalar",
    "smooth_local_distance_difference_test",
    "weighted_rigid_align",
]
