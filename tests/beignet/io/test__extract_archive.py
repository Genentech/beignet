import tarfile
import zipfile
from pathlib import Path

import pytest

from beignet.io import extract_archive


def _make_tar_gz(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content in members.items():
            member_path = path.parent / name
            member_path.parent.mkdir(parents=True, exist_ok=True)
            member_path.write_bytes(content)
            archive.add(member_path, arcname=name)
            member_path.unlink()
            # clean empty nested dirs created for member staging
            parent = member_path.parent
            while (
                parent != path.parent and parent.exists() and not any(parent.iterdir())
            ):
                parent.rmdir()
                parent = parent.parent
    return path


class TestExtractArchive:
    def test_extract_tar_gz(self, tmp_path: Path):
        archive_path = _make_tar_gz(
            tmp_path / "archive.tar.gz",
            {"hello.txt": b"hello", "nested/world.txt": b"world"},
        )
        destination = tmp_path / "out"

        result = extract_archive(archive_path, destination)

        assert result == destination
        assert (destination / "hello.txt").read_bytes() == b"hello"
        assert (destination / "nested" / "world.txt").read_bytes() == b"world"
        assert archive_path.exists()

    def test_extract_zip_and_remove(self, tmp_path: Path):
        archive_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive_path, mode="w") as archive:
            archive.writestr("a.txt", b"A")

        destination = tmp_path / "out"
        result = extract_archive(archive_path, destination, remove_finished=True)

        assert result == destination
        assert (destination / "a.txt").read_bytes() == b"A"
        assert not archive_path.exists()

    def test_unsupported_archive_type(self, tmp_path: Path):
        path = tmp_path / "file.bin"
        path.write_bytes(b"not-an-archive")

        with pytest.raises(RuntimeError, match="Unsupported archive type"):
            extract_archive(path, tmp_path / "out")
