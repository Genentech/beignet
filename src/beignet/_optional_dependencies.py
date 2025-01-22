import importlib
from typing import List


def optional_dependencies(names: List[str], groups: List[str]):
    modules = []

    for name in names:
        try:
            modules = [*modules, importlib.import_module(name)]
        except ImportError:
            modules = [*modules, None]

    missing_names = []

    for name, module in zip(names, modules, strict=False):
        if module is None:
            missing_names = [*missing_names, name]

    if len(missing_names) == 0:
        return

    message = "Missing optional dependencies:\n"

    for missing_name in missing_names:
        message = f"{message}\n    - {missing_name}"

    if len(groups) == 1:
        groups = groups[0]
    else:
        groups = ", ".join(groups)

    message = f"{message}\n\n"

    message = f'{message}Try:\n\n    $ pip install "beignet[{groups}]"'

    raise ImportError(message)
