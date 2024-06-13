import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("beignet")
except PackageNotFoundError:
    __version__ = None

from ._add_chebyshev_series import add_chebyshev_series
from ._add_laguerre_series import add_laguerre_series
from ._add_legendre_series import add_legendre_series
from ._add_physicists_hermite_series import add_physicists_hermite_series
from ._add_power_series import add_power_series
from ._add_probabilists_hermite_series import add_probabilists_hermite_series
from ._apply_euler_angle import apply_euler_angle
from ._apply_quaternion import (
    apply_quaternion,
)
from ._apply_rotation_matrix import apply_rotation_matrix
from ._apply_rotation_vector import apply_rotation_vector
from ._apply_transform import apply_transform
from ._chebyshev_series_to_polynomial_series import (
    chebyshev_series_to_polynomial_series,
)
from ._compose_euler_angle import compose_euler_angle
from ._compose_quaternion import compose_quaternion
from ._compose_rotation_matrix import compose_rotation_matrix
from ._compose_rotation_vector import compose_rotation_vector
from ._differentiate_chebyshev_series import differentiate_chebyshev_series
from ._differentiate_laguerre_series import differentiate_laguerre_series
from ._differentiate_legendre_series import differentiate_legendre_series
from ._differentiate_physicists_hermite_series import (
    differentiate_physicists_hermite_series,
)
from ._differentiate_power_series import differentiate_power_series
from ._differentiate_probabilists_hermite_series import (
    differentiate_probabilists_hermite_series,
)
from ._divide_chebyshev_series import divide_chebyshev_series
from ._divide_laguerre_series import divide_laguerre_series
from ._divide_legendre_series import divide_legendre_series
from ._divide_physicists_hermite_series import divide_physicists_hermite_series
from ._divide_power_series import divide_power_series
from ._divide_probabilists_hermite_series import divide_probabilists_hermite_series
from ._euler_angle_identity import euler_angle_identity
from ._euler_angle_magnitude import euler_angle_magnitude
from ._euler_angle_mean import euler_angle_mean
from ._euler_angle_to_quaternion import (
    euler_angle_to_quaternion,
)
from ._euler_angle_to_rotation_matrix import euler_angle_to_rotation_matrix
from ._euler_angle_to_rotation_vector import euler_angle_to_rotation_vector
from ._fit_chebyshev_series import fit_chebyshev_series
from ._fit_laguerre_series import fit_laguerre_series
from ._fit_legendre_series import fit_legendre_series
from ._fit_physicists_hermite_series import fit_physicists_hermite_series
from ._fit_power_series import fit_power_series
from ._fit_probabilists_hermite_series import fit_probabilists_hermite_series
from ._integrate_chebyshev_series import integrate_chebyshev_series
from ._integrate_laguerre_series import integrate_laguerre_series
from ._integrate_legendre_series import integrate_legendre_series
from ._integrate_physicists_hermite_series import integrate_physicists_hermite_series
from ._integrate_power_series import integrate_power_series
from ._integrate_probabilists_hermite_series import (
    integrate_probabilists_hermite_series,
)
from ._invert_euler_angle import invert_euler_angle
from ._invert_quaternion import invert_quaternion
from ._invert_rotation_matrix import invert_rotation_matrix
from ._invert_rotation_vector import invert_rotation_vector
from ._invert_transform import invert_transform
from ._lennard_jones_potential import lennard_jones_potential
from ._multiply_chebyshev_series import multiply_chebyshev_series
from ._multiply_laguerre_series import multiply_laguerre_series
from ._multiply_legendre_series import multiply_legendre_series
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series
from ._multiply_power_series import multiply_power_series
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series
from ._quaternion_identity import quaternion_identity
from ._quaternion_magnitude import quaternion_magnitude
from ._quaternion_mean import quaternion_mean
from ._quaternion_slerp import quaternion_slerp
from ._quaternion_to_euler_angle import (
    quaternion_to_euler_angle,
)
from ._quaternion_to_rotation_matrix import (
    quaternion_to_rotation_matrix,
)
from ._quaternion_to_rotation_vector import (
    quaternion_to_rotation_vector,
)
from ._random_euler_angle import random_euler_angle
from ._random_quaternion import random_quaternion
from ._random_rotation_matrix import random_rotation_matrix
from ._random_rotation_vector import random_rotation_vector
from ._rotation_matrix_identity import rotation_matrix_identity
from ._rotation_matrix_magnitude import rotation_matrix_magnitude
from ._rotation_matrix_mean import rotation_matrix_mean
from ._rotation_matrix_to_euler_angle import rotation_matrix_to_euler_angle
from ._rotation_matrix_to_quaternion import (
    rotation_matrix_to_quaternion,
)
from ._rotation_matrix_to_rotation_vector import (
    rotation_matrix_to_rotation_vector,
)
from ._rotation_vector_identity import rotation_vector_identity
from ._rotation_vector_magnitude import rotation_vector_magnitude
from ._rotation_vector_mean import rotation_vector_mean
from ._rotation_vector_to_euler_angle import rotation_vector_to_euler_angle
from ._rotation_vector_to_quaternion import (
    rotation_vector_to_quaternion,
)
from ._rotation_vector_to_rotation_matrix import (
    rotation_vector_to_rotation_matrix,
)
from ._subtract_chebyshev_series import subtract_chebyshev_series
from ._subtract_laguerre_series import subtract_laguerre_series
from ._subtract_legendre_series import subtract_legendre_series
from ._subtract_physicists_hermite_series import subtract_physicists_hermite_series
from ._subtract_power_series import subtract_power_series
from ._subtract_probabilists_hermite_series import subtract_probabilists_hermite_series
from ._translation_identity import translation_identity
from ._trim_chebyshev_series import trim_chebyshev_series
from ._trim_laguerre_series import trim_laguerre_series
from ._trim_legendre_series import trim_legendre_series
from ._trim_physicists_hermite_series import trim_physicists_hermite_series
from ._trim_power_series import trim_power_series
from ._trim_probabilists_hermite_series import trim_probabilists_hermite_series
from .special import error_erf, error_erfc

__all__ = [
    "add_chebyshev_series",
    "add_laguerre_series",
    "add_legendre_series",
    "add_physicists_hermite_series",
    "add_power_series",
    "add_probabilists_hermite_series",
    "apply_euler_angle",
    "apply_quaternion",
    "apply_rotation_matrix",
    "apply_rotation_vector",
    "apply_transform",
    "chebyshev_series_to_polynomial_series",
    "compose_euler_angle",
    "compose_quaternion",
    "compose_rotation_matrix",
    "compose_rotation_vector",
    "differentiate_chebyshev_series",
    "differentiate_laguerre_series",
    "differentiate_legendre_series",
    "differentiate_physicists_hermite_series",
    "differentiate_power_series",
    "differentiate_probabilists_hermite_series",
    "divide_chebyshev_series",
    "divide_laguerre_series",
    "divide_legendre_series",
    "divide_physicists_hermite_series",
    "divide_power_series",
    "divide_probabilists_hermite_series",
    "euler_angle_identity",
    "euler_angle_magnitude",
    "euler_angle_mean",
    "euler_angle_to_quaternion",
    "euler_angle_to_rotation_matrix",
    "euler_angle_to_rotation_vector",
    "fit_chebyshev_series",
    "fit_laguerre_series",
    "fit_legendre_series",
    "fit_physicists_hermite_series",
    "fit_power_series",
    "fit_probabilists_hermite_series",
    "integrate_chebyshev_series",
    "integrate_laguerre_series",
    "integrate_legendre_series",
    "integrate_physicists_hermite_series",
    "integrate_power_series",
    "integrate_probabilists_hermite_series",
    "invert_euler_angle",
    "invert_quaternion",
    "invert_rotation_matrix",
    "invert_rotation_vector",
    "invert_transform",
    "lennard_jones_potential",
    "multiply_chebyshev_series",
    "multiply_laguerre_series",
    "multiply_legendre_series",
    "multiply_physicists_hermite_series",
    "multiply_power_series",
    "multiply_probabilists_hermite_series",
    "quaternion_identity",
    "quaternion_magnitude",
    "quaternion_mean",
    "quaternion_slerp",
    "quaternion_to_euler_angle",
    "quaternion_to_rotation_matrix",
    "quaternion_to_rotation_vector",
    "random_euler_angle",
    "random_quaternion",
    "random_rotation_matrix",
    "random_rotation_vector",
    "rotation_matrix_identity",
    "rotation_matrix_magnitude",
    "rotation_matrix_mean",
    "rotation_matrix_to_euler_angle",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_rotation_vector",
    "rotation_vector_identity",
    "rotation_vector_magnitude",
    "rotation_vector_mean",
    "rotation_vector_to_euler_angle",
    "rotation_vector_to_quaternion",
    "rotation_vector_to_rotation_matrix",
    "subtract_legendre_series",
    "subtract_physicists_hermite_series",
    "subtract_power_series",
    "subtract_probabilists_hermite_series",
    "translation_identity",
    "trim_chebyshev_series",
    "trim_laguerre_series",
    "trim_legendre_series",
    "trim_physicists_hermite_series",
    "trim_power_series",
    "trim_probabilists_hermite_series",
]
