from __future__ import annotations

from pathlib import Path

from ._download import download
from ._extract_archive import extract_archive


def download_and_extract_archive(
    resource: str,
    source: str | Path | None = None,
    destination: str | Path | None = None,
    *,
    name: str | None = None,
    checksum: str | None = None,
    remove_archive: bool = False,
) -> Path:
    """
    Download an archive from ``resource`` and extract it.

    Parameters
    ----------
    resource : str
        URL of the archive to download.
    source : str | Path | None
        Directory where the archive is downloaded. If omitted, ``destination``
        is used.
    destination : str | Path | None
        Directory where the archive is extracted. If omitted, ``source`` is
        used.
    name : str | None
        Filename to save the archive under. Defaults to the URL basename.
    checksum : str | None
        Optional checksum of the archive. Bare 32-character hex digests are
        treated as MD5. Otherwise use ``algorithm:hexdigest``.
    remove_archive : bool
        If ``True``, delete the archive after a successful extraction.

    Returns
    -------
    Path
        Directory the archive was extracted into.
    """
    if source is None and destination is None:
        raise ValueError("Either source or destination must be provided")

    if source is None:
        source = destination

    if destination is None:
        destination = source

    archive_path = download(
        source=resource,
        destination=source,
        filename=name,
        checksum=checksum,
    )

    return extract_archive(
        archive_path,
        destination,
        remove_finished=remove_archive,
    )
