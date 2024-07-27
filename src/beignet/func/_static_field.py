import dataclasses


def static_field():
    return dataclasses.field(metadata={"static": True})
