import dataclasses

from optree import PyTree

from ..__dataclass import _dataclass
from .__parameter_tree_kind import _ParameterTreeKind


@_dataclass
class _ParameterTree:
    tree: PyTree
    kind: _ParameterTreeKind = dataclasses.field(metadata={"static": True})
