import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("beignet")
except PackageNotFoundError:
    __version__ = None

from ._apply_rotation_matrix import apply_rotation_matrix
from ._apply_transform import apply_transform
from ._invert_rotation_matrix import invert_rotation_matrix
from ._invert_transform import invert_transform

__all__ = [
    "apply_rotation_matrix",
    "apply_transform",
    "invert_rotation_matrix",
    "invert_transform",
]
