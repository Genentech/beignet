import dataclasses
from typing import List, Tuple, Type, TypeVar

import optree

T = TypeVar("T")


def _dataclass(cls: Type[T]):
    def _set(self: dataclasses.dataclass, **kwargs):
        return dataclasses.replace(self, **kwargs)

    cls.set = _set

    dataclass_cls = dataclasses.dataclass(frozen=True)(cls)

    data_fields, metadata_fields = [], []

    for name, kind in dataclass_cls.__dataclass_fields__.items():
        if not kind.metadata.get("static", False):
            data_fields = [*data_fields, name]
        else:
            metadata_fields = [*metadata_fields, name]

    def _iterate_cls(_x) -> List[Tuple]:
        data_iterable = []

        for k in data_fields:
            data_iterable.append(getattr(_x, k))

        metadata_iterable = []

        for k in metadata_fields:
            metadata_iterable.append(getattr(_x, k))

        return [data_iterable, metadata_iterable]

    def _iterable_to_cls(meta, data):
        meta_args = tuple(zip(metadata_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)

        return dataclass_cls(**kwargs)

    optree.register_pytree_node(
        dataclass_cls,
        _iterate_cls,
        _iterable_to_cls,
        "prescient.func",
    )

    return dataclass_cls
