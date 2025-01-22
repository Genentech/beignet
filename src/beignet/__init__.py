import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("beignet")
except PackageNotFoundError:
    __version__ = None

from ._add_chebyshev_polynomial import add_chebyshev_polynomial
from ._add_laguerre_polynomial import add_laguerre_polynomial
from ._add_legendre_polynomial import add_legendre_polynomial
from ._add_physicists_hermite_polynomial import add_physicists_hermite_polynomial
from ._add_polynomial import add_polynomial
from ._add_probabilists_hermite_polynomial import add_probabilists_hermite_polynomial
from ._apply_euler_angle import apply_euler_angle
from ._apply_quaternion import (
    apply_quaternion,
)
from ._apply_rotation_matrix import apply_rotation_matrix
from ._apply_rotation_vector import apply_rotation_vector
from ._apply_transform import apply_transform
from ._chebyshev_extrema import chebyshev_extrema
from ._chebyshev_gauss_quadrature import chebyshev_gauss_quadrature
from ._chebyshev_interpolation import chebyshev_interpolation
from ._chebyshev_polynomial_companion import chebyshev_polynomial_companion
from ._chebyshev_polynomial_domain import chebyshev_polynomial_domain
from ._chebyshev_polynomial_from_roots import chebyshev_polynomial_from_roots
from ._chebyshev_polynomial_one import chebyshev_polynomial_one
from ._chebyshev_polynomial_power import chebyshev_polynomial_power
from ._chebyshev_polynomial_roots import chebyshev_polynomial_roots
from ._chebyshev_polynomial_to_polynomial import chebyshev_polynomial_to_polynomial
from ._chebyshev_polynomial_vandermonde import chebyshev_polynomial_vandermonde
from ._chebyshev_polynomial_vandermonde_2d import chebyshev_polynomial_vandermonde_2d
from ._chebyshev_polynomial_vandermonde_3d import chebyshev_polynomial_vandermonde_3d
from ._chebyshev_polynomial_weight import chebyshev_polynomial_weight
from ._chebyshev_polynomial_x import chebyshev_polynomial_x
from ._chebyshev_polynomial_zero import chebyshev_polynomial_zero
from ._chebyshev_zeros import chebyshev_zeros
from ._compose_euler_angle import compose_euler_angle
from ._compose_quaternion import compose_quaternion
from ._compose_rotation_matrix import compose_rotation_matrix
from ._compose_rotation_vector import compose_rotation_vector
from ._convolve import convolve
from ._differentiate_chebyshev_polynomial import differentiate_chebyshev_polynomial
from ._differentiate_laguerre_polynomial import differentiate_laguerre_polynomial
from ._differentiate_legendre_polynomial import differentiate_legendre_polynomial
from ._differentiate_physicists_hermite_polynomial import (
    differentiate_physicists_hermite_polynomial,
)
from ._differentiate_polynomial import differentiate_polynomial
from ._differentiate_probabilists_hermite_polynomial import (
    differentiate_probabilists_hermite_polynomial,
)
from ._divide_chebyshev_polynomial import divide_chebyshev_polynomial
from ._divide_laguerre_polynomial import divide_laguerre_polynomial
from ._divide_legendre_polynomial import divide_legendre_polynomial
from ._divide_physicists_hermite_polynomial import divide_physicists_hermite_polynomial
from ._divide_polynomial import divide_polynomial
from ._divide_probabilists_hermite_polynomial import (
    divide_probabilists_hermite_polynomial,
)
from ._euler_angle_identity import euler_angle_identity
from ._euler_angle_magnitude import euler_angle_magnitude
from ._euler_angle_mean import euler_angle_mean
from ._euler_angle_to_quaternion import (
    euler_angle_to_quaternion,
)
from ._euler_angle_to_rotation_matrix import euler_angle_to_rotation_matrix
from ._euler_angle_to_rotation_vector import euler_angle_to_rotation_vector
from ._evaluate_chebyshev_polynomial import evaluate_chebyshev_polynomial
from ._evaluate_chebyshev_polynomial_2d import evaluate_chebyshev_polynomial_2d
from ._evaluate_chebyshev_polynomial_3d import evaluate_chebyshev_polynomial_3d
from ._evaluate_chebyshev_polynomial_cartesian_2d import (
    evaluate_chebyshev_polynomial_cartesian_2d,
)
from ._evaluate_chebyshev_polynomial_cartesian_3d import (
    evaluate_chebyshev_polynomial_cartesian_3d,
)
from ._evaluate_laguerre_polynomial import evaluate_laguerre_polynomial
from ._evaluate_laguerre_polynomial_2d import evaluate_laguerre_polynomial_2d
from ._evaluate_laguerre_polynomial_3d import evaluate_laguerre_polynomial_3d
from ._evaluate_laguerre_polynomial_cartesian_2d import (
    evaluate_laguerre_polynomial_cartesian_2d,
)
from ._evaluate_laguerre_polynomial_cartesian_3d import (
    evaluate_laguerre_polynomial_cartesian_3d,
)
from ._evaluate_legendre_polynomial import evaluate_legendre_polynomial
from ._evaluate_legendre_polynomial_2d import evaluate_legendre_polynomial_2d
from ._evaluate_legendre_polynomial_3d import evaluate_legendre_polynomial_3d
from ._evaluate_legendre_polynomial_cartesian_2d import (
    evaluate_legendre_polynomial_cartesian_2d,
)
from ._evaluate_legendre_polynomial_cartesian_3d import (
    evaluate_legendre_polynomial_cartesian_3d,
)
from ._evaluate_physicists_hermite_polynomial import (
    evaluate_physicists_hermite_polynomial,
)
from ._evaluate_physicists_hermite_polynomial_2d import (
    evaluate_physicists_hermite_polynomial_2d,
)
from ._evaluate_physicists_hermite_polynomial_3d import (
    evaluate_physicists_hermite_polynomial_3d,
)
from ._evaluate_physicists_hermite_polynomial_cartesian_2d import (
    evaluate_physicists_hermite_polynomial_cartesian_2d,
)
from ._evaluate_physicists_hermite_polynomial_cartesian_3d import (
    evaluate_physicists_hermite_polynomial_cartesian_3d,
)
from ._evaluate_polynomial import evaluate_polynomial
from ._evaluate_polynomial_2d import evaluate_polynomial_2d
from ._evaluate_polynomial_3d import evaluate_polynomial_3d
from ._evaluate_polynomial_cartesian_2d import evaluate_polynomial_cartesian_2d
from ._evaluate_polynomial_cartesian_3d import evaluate_polynomial_cartesian_3d
from ._evaluate_polynomial_from_roots import evaluate_polynomial_from_roots
from ._evaluate_probabilists_hermite_polynomial import (
    evaluate_probabilists_hermite_polynomial,
)
from ._evaluate_probabilists_hermite_polynomial_2d import (
    evaluate_probabilists_hermite_polynomial_2d,
)
from ._evaluate_probabilists_hermite_polynomial_3d import (
    evaluate_probabilists_hermite_polynomial_3d,
)
from ._evaluate_probabilists_hermite_polynomial_cartersian_2d import (
    evaluate_probabilists_hermite_polynomial_cartersian_2d,
)
from ._evaluate_probabilists_hermite_polynomial_cartersian_3d import (
    evaluate_probabilists_hermite_polynomial_cartersian_3d,
)
from ._farthest_first_traversal import farthest_first_traversal
from ._fit_chebyshev_polynomial import fit_chebyshev_polynomial
from ._fit_laguerre_polynomial import fit_laguerre_polynomial
from ._fit_legendre_polynomial import fit_legendre_polynomial
from ._fit_physicists_hermite_polynomial import fit_physicists_hermite_polynomial
from ._fit_polynomial import fit_polynomial
from ._fit_probabilists_hermite_polynomial import fit_probabilists_hermite_polynomial
from ._gauss_laguerre_quadrature import gauss_laguerre_quadrature
from ._gauss_legendre_quadrature import gauss_legendre_quadrature
from ._gauss_physicists_hermite_polynomial_quadrature import (
    gauss_physicists_hermite_polynomial_quadrature,
)
from ._gauss_probabilists_hermite_polynomial_quadrature import (
    gauss_probabilists_hermite_polynomial_quadrature,
)
from ._integrate_chebyshev_polynomial import integrate_chebyshev_polynomial
from ._integrate_laguerre_polynomial import integrate_laguerre_polynomial
from ._integrate_legendre_polynomial import integrate_legendre_polynomial
from ._integrate_physicists_hermite_polynomial import (
    integrate_physicists_hermite_polynomial,
)
from ._integrate_polynomial import integrate_polynomial
from ._integrate_probabilists_hermite_polynomial import (
    integrate_probabilists_hermite_polynomial,
)
from ._invert_euler_angle import invert_euler_angle
from ._invert_quaternion import invert_quaternion
from ._invert_rotation_matrix import invert_rotation_matrix
from ._invert_rotation_vector import invert_rotation_vector
from ._invert_transform import invert_transform
from ._laguerre_polynomial_companion import laguerre_polynomial_companion
from ._laguerre_polynomial_domain import laguerre_polynomial_domain
from ._laguerre_polynomial_from_roots import laguerre_polynomial_from_roots
from ._laguerre_polynomial_one import laguerre_polynomial_one
from ._laguerre_polynomial_power import laguerre_polynomial_power
from ._laguerre_polynomial_roots import laguerre_polynomial_roots
from ._laguerre_polynomial_to_polynomial import laguerre_polynomial_to_polynomial
from ._laguerre_polynomial_vandermonde import laguerre_polynomial_vandermonde
from ._laguerre_polynomial_vandermonde_2d import laguerre_polynomial_vandermonde_2d
from ._laguerre_polynomial_vandermonde_3d import laguerre_polynomial_vandermonde_3d
from ._laguerre_polynomial_weight import laguerre_polynomial_weight
from ._laguerre_polynomial_x import laguerre_polynomial_x
from ._laguerre_polynomial_zero import laguerre_polynomial_zero
from ._legendre_polynomial_companion import legendre_polynomial_companion
from ._legendre_polynomial_domain import legendre_polynomial_domain
from ._legendre_polynomial_from_roots import legendre_polynomial_from_roots
from ._legendre_polynomial_one import legendre_polynomial_one
from ._legendre_polynomial_power import legendre_polynomial_power
from ._legendre_polynomial_roots import legendre_polynomial_roots
from ._legendre_polynomial_to_polynomial import legendre_polynomial_to_polynomial
from ._legendre_polynomial_vandermonde import legendre_polynomial_vandermonde
from ._legendre_polynomial_vandermonde_2d import legendre_polynomial_vandermonde_2d
from ._legendre_polynomial_vandermonde_3d import legendre_polynomial_vandermonde_3d
from ._legendre_polynomial_weight import legendre_polynomial_weight
from ._legendre_polynomial_x import legendre_polynomial_x
from ._legendre_polynomial_zero import legendre_polynomial_zero
from ._lennard_jones_potential import lennard_jones_potential
from ._linear_chebyshev_polynomial import linear_chebyshev_polynomial
from ._linear_laguerre_polynomial import linear_laguerre_polynomial
from ._linear_legendre_polynomial import linear_legendre_polynomial
from ._linear_physicists_hermite_polynomial import linear_physicists_hermite_polynomial
from ._linear_polynomial import linear_polynomial
from ._linear_probabilists_hermite_polynomial import (
    linear_probabilists_hermite_polynomial,
)
from ._multiply_chebyshev_polynomial import multiply_chebyshev_polynomial
from ._multiply_chebyshev_polynomial_by_x import multiply_chebyshev_polynomial_by_x
from ._multiply_laguerre_polynomial import multiply_laguerre_polynomial
from ._multiply_laguerre_polynomial_by_x import multiply_laguerre_polynomial_by_x
from ._multiply_legendre_polynomial import multiply_legendre_polynomial
from ._multiply_legendre_polynomial_by_x import multiply_legendre_polynomial_by_x
from ._multiply_physicists_hermite_polynomial import (
    multiply_physicists_hermite_polynomial,
)
from ._multiply_physicists_hermite_polynomial_by_x import (
    multiply_physicists_hermite_polynomial_by_x,
)
from ._multiply_polynomial import multiply_polynomial
from ._multiply_polynomial_by_x import multiply_polynomial_by_x
from ._multiply_probabilists_hermite_polynomial import (
    multiply_probabilists_hermite_polynomial,
)
from ._multiply_probabilists_hermite_polynomial_by_x import (
    multiply_probabilists_hermite_polynomial_by_x,
)
from ._optional_dependencies import optional_dependencies
from ._physicists_hermite_polynomial_companion import (
    physicists_hermite_polynomial_companion,
)
from ._physicists_hermite_polynomial_domain import physicists_hermite_polynomial_domain
from ._physicists_hermite_polynomial_from_roots import (
    physicists_hermite_polynomial_from_roots,
)
from ._physicists_hermite_polynomial_one import physicists_hermite_polynomial_one
from ._physicists_hermite_polynomial_power import physicists_hermite_polynomial_power
from ._physicists_hermite_polynomial_roots import physicists_hermite_polynomial_roots
from ._physicists_hermite_polynomial_to_polynomial import (
    physicists_hermite_polynomial_to_polynomial,
)
from ._physicists_hermite_polynomial_vandermonde import (
    physicists_hermite_polynomial_vandermonde,
)
from ._physicists_hermite_polynomial_vandermonde_2d import (
    physicists_hermite_polynomial_vandermonde_2d,
)
from ._physicists_hermite_polynomial_vandermonde_3d import (
    physicists_hermite_polynomial_vandermonde_3d,
)
from ._physicists_hermite_polynomial_weight import physicists_hermite_polynomial_weight
from ._physicists_hermite_polynomial_x import physicists_hermite_polynomial_x
from ._physicists_hermite_polynomial_zero import physicists_hermite_polynomial_zero
from ._polynomial_companion import polynomial_companion
from ._polynomial_domain import polynomial_domain
from ._polynomial_from_roots import polynomial_from_roots
from ._polynomial_one import polynomial_one
from ._polynomial_power import polynomial_power
from ._polynomial_roots import polynomial_roots
from ._polynomial_to_chebyshev_polynomial import polynomial_to_chebyshev_polynomial
from ._polynomial_to_laguerre_polynomial import polynomial_to_laguerre_polynomial
from ._polynomial_to_legendre_polynomial import polynomial_to_legendre_polynomial
from ._polynomial_to_physicists_hermite_polynomial import (
    polynomial_to_physicists_hermite_polynomial,
)
from ._polynomial_to_probabilists_hermite_polynomial import (
    polynomial_to_probabilists_hermite_polynomial,
)
from ._polynomial_vandermonde import polynomial_vandermonde
from ._polynomial_vandermonde_2d import polynomial_vandermonde_2d
from ._polynomial_vandermonde_3d import polynomial_vandermonde_3d
from ._polynomial_x import polynomial_x
from ._polynomial_zero import polynomial_zero
from ._probabilists_hermite_polynomial_companion import (
    probabilists_hermite_polynomial_companion,
)
from ._probabilists_hermite_polynomial_domain import (
    probabilists_hermite_polynomial_domain,
)
from ._probabilists_hermite_polynomial_from_roots import (
    probabilists_hermite_polynomial_from_roots,
)
from ._probabilists_hermite_polynomial_one import probabilists_hermite_polynomial_one
from ._probabilists_hermite_polynomial_power import (
    probabilists_hermite_polynomial_power,
)
from ._probabilists_hermite_polynomial_roots import (
    probabilists_hermite_polynomial_roots,
)
from ._probabilists_hermite_polynomial_to_polynomial import (
    probabilists_hermite_polynomial_to_polynomial,
)
from ._probabilists_hermite_polynomial_vandermonde import (
    probabilists_hermite_polynomial_vandermonde,
)
from ._probabilists_hermite_polynomial_vandermonde_2d import (
    probabilists_hermite_polynomial_vandermonde_2d,
)
from ._probabilists_hermite_polynomial_vandermonde_3d import (
    probabilists_hermite_polynomial_vandermonde_3d,
)
from ._probabilists_hermite_polynomial_weight import (
    probabilists_hermite_polynomial_weight,
)
from ._probabilists_hermite_polynomial_x import probabilists_hermite_polynomial_x
from ._probabilists_hermite_polynomial_zero import probabilists_hermite_polynomial_zero
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
from ._subtract_chebyshev_polynomial import subtract_chebyshev_polynomial
from ._subtract_laguerre_polynomial import subtract_laguerre_polynomial
from ._subtract_legendre_polynomial import subtract_legendre_polynomial
from ._subtract_physicists_hermite_polynomial import (
    subtract_physicists_hermite_polynomial,
)
from ._subtract_polynomial import subtract_polynomial
from ._subtract_probabilists_hermite_polynomial import (
    subtract_probabilists_hermite_polynomial,
)
from ._translation_identity import translation_identity
from ._trim_chebyshev_polynomial_coefficients import (
    trim_chebyshev_polynomial_coefficients,
)
from ._trim_laguerre_polynomial_coefficients import (
    trim_laguerre_polynomial_coefficients,
)
from ._trim_legendre_polynomial_coefficients import (
    trim_legendre_polynomial_coefficients,
)
from ._trim_physicists_hermite_polynomial_coefficients import (
    trim_physicists_hermite_polynomial_coefficients,
)
from ._trim_polynomial_coefficients import trim_polynomial_coefficients
from ._trim_probabilists_hermite_polynomial_coefficients import (
    trim_probabilists_hermite_polynomial_coefficients,
)
from .special import error_erf, error_erfc

__all__ = [
    "add_chebyshev_polynomial",
    "add_laguerre_polynomial",
    "add_legendre_polynomial",
    "add_physicists_hermite_polynomial",
    "add_polynomial",
    "add_probabilists_hermite_polynomial",
    "apply_euler_angle",
    "apply_quaternion",
    "apply_rotation_matrix",
    "apply_rotation_vector",
    "apply_transform",
    "chebyshev_extrema",
    "chebyshev_gauss_quadrature",
    "chebyshev_interpolation",
    "chebyshev_polynomial_companion",
    "chebyshev_polynomial_domain",
    "chebyshev_polynomial_from_roots",
    "chebyshev_polynomial_one",
    "chebyshev_polynomial_power",
    "chebyshev_polynomial_roots",
    "chebyshev_polynomial_to_polynomial",
    "chebyshev_polynomial_vandermonde",
    "chebyshev_polynomial_vandermonde_2d",
    "chebyshev_polynomial_vandermonde_3d",
    "chebyshev_polynomial_weight",
    "chebyshev_polynomial_x",
    "chebyshev_polynomial_zero",
    "chebyshev_zeros",
    "compose_euler_angle",
    "compose_quaternion",
    "compose_rotation_matrix",
    "compose_rotation_vector",
    "convolve",
    "differentiate_chebyshev_polynomial",
    "differentiate_laguerre_polynomial",
    "differentiate_legendre_polynomial",
    "differentiate_physicists_hermite_polynomial",
    "differentiate_polynomial",
    "differentiate_probabilists_hermite_polynomial",
    "divide_chebyshev_polynomial",
    "divide_laguerre_polynomial",
    "divide_legendre_polynomial",
    "divide_physicists_hermite_polynomial",
    "divide_polynomial",
    "divide_probabilists_hermite_polynomial",
    "euler_angle_identity",
    "euler_angle_magnitude",
    "euler_angle_mean",
    "euler_angle_to_quaternion",
    "euler_angle_to_rotation_matrix",
    "euler_angle_to_rotation_vector",
    "evaluate_chebyshev_polynomial",
    "evaluate_chebyshev_polynomial_2d",
    "evaluate_chebyshev_polynomial_3d",
    "evaluate_chebyshev_polynomial_cartesian_2d",
    "evaluate_chebyshev_polynomial_cartesian_3d",
    "evaluate_laguerre_polynomial",
    "evaluate_laguerre_polynomial_2d",
    "evaluate_laguerre_polynomial_3d",
    "evaluate_laguerre_polynomial_cartesian_2d",
    "evaluate_laguerre_polynomial_cartesian_3d",
    "evaluate_legendre_polynomial",
    "evaluate_legendre_polynomial_2d",
    "evaluate_legendre_polynomial_3d",
    "evaluate_legendre_polynomial_cartesian_2d",
    "evaluate_legendre_polynomial_cartesian_3d",
    "evaluate_physicists_hermite_polynomial",
    "evaluate_physicists_hermite_polynomial_2d",
    "evaluate_physicists_hermite_polynomial_3d",
    "evaluate_physicists_hermite_polynomial_cartesian_2d",
    "evaluate_physicists_hermite_polynomial_cartesian_3d",
    "evaluate_polynomial",
    "evaluate_polynomial_2d",
    "evaluate_polynomial_3d",
    "evaluate_polynomial_cartesian_2d",
    "evaluate_polynomial_cartesian_3d",
    "evaluate_polynomial_from_roots",
    "evaluate_probabilists_hermite_polynomial",
    "evaluate_probabilists_hermite_polynomial_2d",
    "evaluate_probabilists_hermite_polynomial_3d",
    "evaluate_probabilists_hermite_polynomial_cartersian_2d",
    "evaluate_probabilists_hermite_polynomial_cartersian_3d",
    "farthest_first_traversal",
    "fit_chebyshev_polynomial",
    "fit_laguerre_polynomial",
    "fit_legendre_polynomial",
    "fit_physicists_hermite_polynomial",
    "fit_polynomial",
    "fit_probabilists_hermite_polynomial",
    "gauss_laguerre_quadrature",
    "gauss_legendre_quadrature",
    "gauss_physicists_hermite_polynomial_quadrature",
    "gauss_probabilists_hermite_polynomial_quadrature",
    "integrate_chebyshev_polynomial",
    "integrate_laguerre_polynomial",
    "integrate_legendre_polynomial",
    "integrate_physicists_hermite_polynomial",
    "integrate_polynomial",
    "integrate_probabilists_hermite_polynomial",
    "invert_euler_angle",
    "invert_quaternion",
    "invert_rotation_matrix",
    "invert_rotation_vector",
    "invert_transform",
    "laguerre_polynomial_companion",
    "laguerre_polynomial_domain",
    "laguerre_polynomial_from_roots",
    "laguerre_polynomial_one",
    "laguerre_polynomial_power",
    "laguerre_polynomial_roots",
    "laguerre_polynomial_to_polynomial",
    "laguerre_polynomial_vandermonde",
    "laguerre_polynomial_vandermonde_2d",
    "laguerre_polynomial_vandermonde_3d",
    "laguerre_polynomial_weight",
    "laguerre_polynomial_x",
    "laguerre_polynomial_zero",
    "legendre_polynomial_companion",
    "legendre_polynomial_domain",
    "legendre_polynomial_from_roots",
    "legendre_polynomial_one",
    "legendre_polynomial_power",
    "legendre_polynomial_roots",
    "legendre_polynomial_to_polynomial",
    "legendre_polynomial_vandermonde",
    "legendre_polynomial_vandermonde_2d",
    "legendre_polynomial_vandermonde_3d",
    "legendre_polynomial_weight",
    "legendre_polynomial_x",
    "legendre_polynomial_zero",
    "lennard_jones_potential",
    "linear_chebyshev_polynomial",
    "linear_laguerre_polynomial",
    "linear_legendre_polynomial",
    "linear_physicists_hermite_polynomial",
    "linear_polynomial",
    "linear_probabilists_hermite_polynomial",
    "multiply_chebyshev_polynomial",
    "multiply_chebyshev_polynomial_by_x",
    "multiply_laguerre_polynomial",
    "multiply_laguerre_polynomial_by_x",
    "multiply_legendre_polynomial",
    "multiply_legendre_polynomial_by_x",
    "multiply_physicists_hermite_polynomial",
    "multiply_physicists_hermite_polynomial_by_x",
    "multiply_polynomial",
    "multiply_polynomial_by_x",
    "multiply_probabilists_hermite_polynomial",
    "multiply_probabilists_hermite_polynomial_by_x",
    "optional_dependencies",
    "physicists_hermite_polynomial_companion",
    "physicists_hermite_polynomial_domain",
    "physicists_hermite_polynomial_from_roots",
    "physicists_hermite_polynomial_one",
    "physicists_hermite_polynomial_power",
    "physicists_hermite_polynomial_roots",
    "physicists_hermite_polynomial_to_polynomial",
    "physicists_hermite_polynomial_vandermonde",
    "physicists_hermite_polynomial_vandermonde_2d",
    "physicists_hermite_polynomial_vandermonde_3d",
    "physicists_hermite_polynomial_weight",
    "physicists_hermite_polynomial_x",
    "physicists_hermite_polynomial_zero",
    "polynomial_companion",
    "polynomial_domain",
    "polynomial_from_roots",
    "polynomial_one",
    "polynomial_power",
    "polynomial_roots",
    "polynomial_to_chebyshev_polynomial",
    "polynomial_to_laguerre_polynomial",
    "polynomial_to_legendre_polynomial",
    "polynomial_to_physicists_hermite_polynomial",
    "polynomial_to_probabilists_hermite_polynomial",
    "polynomial_vandermonde",
    "polynomial_vandermonde_2d",
    "polynomial_vandermonde_3d",
    "polynomial_x",
    "polynomial_zero",
    "probabilists_hermite_polynomial_companion",
    "probabilists_hermite_polynomial_domain",
    "probabilists_hermite_polynomial_from_roots",
    "probabilists_hermite_polynomial_one",
    "probabilists_hermite_polynomial_power",
    "probabilists_hermite_polynomial_roots",
    "probabilists_hermite_polynomial_to_polynomial",
    "probabilists_hermite_polynomial_vandermonde",
    "probabilists_hermite_polynomial_vandermonde_2d",
    "probabilists_hermite_polynomial_vandermonde_3d",
    "probabilists_hermite_polynomial_weight",
    "probabilists_hermite_polynomial_x",
    "probabilists_hermite_polynomial_zero",
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
    "subtract_chebyshev_polynomial",
    "subtract_laguerre_polynomial",
    "subtract_legendre_polynomial",
    "subtract_physicists_hermite_polynomial",
    "subtract_polynomial",
    "subtract_probabilists_hermite_polynomial",
    "translation_identity",
    "trim_chebyshev_polynomial_coefficients",
    "trim_laguerre_polynomial_coefficients",
    "trim_legendre_polynomial_coefficients",
    "trim_physicists_hermite_polynomial_coefficients",
    "trim_polynomial_coefficients",
    "trim_probabilists_hermite_polynomial_coefficients",
]
