from typing import Dict


def _merge_dictionaries(
    this: Dict,
    that: Dict,
    ignore_unused_parameters: bool = False,
):
    if not ignore_unused_parameters:
        return {**this, **that}

    merged_dictionaries = dict(this)

    for this_key in merged_dictionaries.keys():
        that_value = that.get(this_key)

        if that_value is not None:
            merged_dictionaries[this_key] = that_value

    return merged_dictionaries
