from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from tqdm.auto import tqdm


def _normalize_checksum(checksum: str | None) -> tuple[str, str] | None:
    if checksum is None:
        return None

    if ":" in checksum:
        algorithm, digest = checksum.split(":", 1)
        return algorithm.lower(), digest.lower()

    checksum = checksum.lower()

    if len(checksum) == 32:
        return "md5", checksum

    return "sha256", checksum


def _file_digest(path: Path, algorithm: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.new(algorithm, usedforsecurity=False)

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def check_integrity(path: str | Path, checksum: str | None = None) -> bool:
    path = Path(path)

    if not path.is_file():
        return False

    normalized = _normalize_checksum(checksum)

    if normalized is None:
        return True

    algorithm, digest = normalized

    return _file_digest(path, algorithm) == digest


def _infer_filename(source: str) -> str:
    filename = Path(urlparse(source).path).name

    if not filename:
        raise ValueError(f"Could not infer filename from source: {source!r}")

    return filename


def _download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".partial")

    try:
        with urllib.request.urlopen(url) as response:
            total = response.length or response.headers.get("Content-Length")
            total = int(total) if total else None

            with (
                temporary_path.open("wb") as file,
                tqdm.wrapattr(
                    file,
                    "write",
                    total=total,
                    desc=path.name,
                    miniters=1,
                ) as output,
            ):
                shutil.copyfileobj(response, output)

        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _download_s3(url: str, path: Path) -> None:
    try:
        import boto3
    except ImportError as error:
        raise ImportError(
            "Downloading from s3:// URLs requires boto3. Try: pip install boto3"
        ) from error

    parsed = urlparse(url)

    if not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URL: {url!r}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".partial")

    try:
        client = boto3.client("s3")
        client.download_file(bucket, key, str(temporary_path))
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def download(
    source: str,
    destination: str | Path,
    *,
    filename: str | None = None,
    checksum: str | None = None,
) -> Path:
    """
    Download ``source`` into ``destination``.

    Parameters
    ----------
    source : str
        URL of the file to download (``http://``, ``https://``, ``ftp://``,
        or ``s3://``).
    destination : str | Path
        Directory in which to place the downloaded file.
    filename : str | None
        Name to save the file under. Defaults to the URL basename.
    checksum : str | None
        Optional checksum of the file. Bare 32-character hex digests are
        treated as MD5. Otherwise use ``algorithm:hexdigest`` (for example
        ``md5:...`` or ``sha256:...``).

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = _infer_filename(source)

    path = destination / filename

    if check_integrity(path, checksum):
        return path

    if source.startswith("s3://"):
        _download_s3(source, path)
    elif source.startswith(("http://", "https://", "ftp://")):
        _download_url(source, path)
    else:
        raise ValueError(f"Unsupported download source: {source!r}")

    if checksum is not None and not check_integrity(path, checksum):
        path.unlink(missing_ok=True)

        raise RuntimeError(
            f"Downloaded file '{path}' failed integrity check "
            f"for checksum {checksum!r}."
        )

    return path
