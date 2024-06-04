import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("beignet")
except PackageNotFoundError:
    __version__ = None

from ._apply_euler_angle import apply_euler_angle
from ._apply_quaternion import (
    apply_quaternion,
)
from ._apply_rotation_matrix import apply_rotation_matrix
from ._apply_rotation_vector import apply_rotation_vector
from ._compose_euler_angle import compose_euler_angle
from ._compose_quaternion import compose_quaternion
from ._compose_rotation_matrix import compose_rotation_matrix
from ._compose_rotation_vector import compose_rotation_vector
from ._euler_angle_identity import euler_angle_identity
from ._euler_angle_magnitude import euler_angle_magnitude
from ._euler_angle_mean import euler_angle_mean
from ._euler_angle_to_quaternion import (
    euler_angle_to_quaternion,
)
from ._euler_angle_to_rotation_matrix import euler_angle_to_rotation_matrix
from ._euler_angle_to_rotation_vector import euler_angle_to_rotation_vector
from ._invert_euler_angle import invert_euler_angle
from ._invert_quaternion import invert_quaternion
from ._invert_rotation_matrix import invert_rotation_matrix
from ._invert_rotation_vector import invert_rotation_vector
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
from ._translation_identity import translation_identity
from .special import erf, erfc

__all__ = [
    "apply_euler_angle",
    "apply_quaternion",
    "apply_rotation_matrix",
    "apply_rotation_vector",
    "compose_euler_angle",
    "compose_quaternion",
    "compose_rotation_matrix",
    "compose_rotation_vector",
    "euler_angle_identity",
    "euler_angle_magnitude",
    "euler_angle_mean",
    "euler_angle_to_quaternion",
    "euler_angle_to_rotation_matrix",
    "euler_angle_to_rotation_vector",
    "invert_euler_angle",
    "invert_quaternion",
    "invert_rotation_matrix",
    "invert_rotation_vector",
    "quaternion_identity",
    "quaternion_magnitude",
    "quaternion_mean",
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
    "quaternion_slerp",
    "translation_identity",
]
