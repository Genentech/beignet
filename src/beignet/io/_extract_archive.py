from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def extract_archive(
    from_path: str | Path,
    to_path: str | Path | None = None,
    *,
    remove_finished: bool = False,
) -> Path:
    """
    Extract an archive to ``to_path``.

    Parameters
    ----------
    from_path : str | Path
        Path to the archive file.
    to_path : str | Path | None
        Directory to extract into. Defaults to the archive's parent directory.
    remove_finished : bool
        If ``True``, delete the archive after a successful extraction.

    Returns
    -------
    Path
        Directory the archive was extracted into.
    """
    from_path = Path(from_path)

    if to_path is None:
        to_path = from_path.parent
    else:
        to_path = Path(to_path)

    to_path.mkdir(parents=True, exist_ok=True)

    name = from_path.name.lower()

    if name.endswith(
        (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    ):
        with tarfile.open(from_path, mode="r:*") as archive:
            # filter="data" is the Python 3.12+ safe default; ignore on older runtimes.
            try:
                archive.extractall(path=to_path, filter="data")
            except TypeError:
                archive.extractall(path=to_path)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(from_path, mode="r") as archive:
            archive.extractall(path=to_path)
    else:
        raise RuntimeError(
            f"Unsupported archive type for '{from_path}'. "
            "Supported suffixes: .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, "
            ".tar.xz, .txz, .zip."
        )

    if remove_finished:
        from_path.unlink()

    return to_path
