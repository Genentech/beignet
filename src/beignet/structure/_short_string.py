def short_string_to_int(input: str):
    """Convert an ascii string with length <= 8 to a uint64 integer."""
    assert input.isascii()
    assert len(input) <= 8
    return int.from_bytes(
        input.ljust(8, "\0").encode("ascii"), byteorder="little", signed=False
    )


def int_to_short_string(input: int):
    assert 0 <= input < 2**64
    """Convert a uint64 integer to an ascii string."""
    return (
        input.to_bytes(length=8, byteorder="little", signed=False)
        .decode("ascii")
        .rstrip("\0")
    )
