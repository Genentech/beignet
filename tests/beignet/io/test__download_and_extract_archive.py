import hashlib
import tarfile
from pathlib import Path
from unittest.mock import patch

from beignet.io import download_and_extract_archive


def _make_tar_gz(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content in members.items():
            member_path = path.parent / name
            member_path.parent.mkdir(parents=True, exist_ok=True)
            member_path.write_bytes(content)
            archive.add(member_path, arcname=name)
            member_path.unlink()
    return path


class TestDownloadAndExtractArchive:
    def test_atom3d_style_positional_args(self, tmp_path: Path):
        """ATOM3DDataset passes (resource, destination, checksum=...)."""
        archive_bytes_path = tmp_path / "remote" / "MSP-raw.tar.gz"
        _make_tar_gz(archive_bytes_path, {"raw/MSP/data/x.bin": b"atom3d"})
        archive_bytes = archive_bytes_path.read_bytes()
        checksum = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()

        destination = tmp_path / "ATOM3DMSP"

        def fake_download(source, destination, *, filename=None, checksum=None):
            assert source == "https://zenodo.org/record/4962515/files/MSP-raw.tar.gz"
            assert filename is None
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "MSP-raw.tar.gz"
            path.write_bytes(archive_bytes)
            return path

        with patch(
            "beignet.io._download_and_extract_archive.download",
            side_effect=fake_download,
        ):
            result = download_and_extract_archive(
                "https://zenodo.org/record/4962515/files/MSP-raw.tar.gz",
                destination,
                checksum=checksum,
            )

        assert result == destination
        assert (
            destination / "raw" / "MSP" / "data" / "x.bin"
        ).read_bytes() == b"atom3d"
        assert (destination / "MSP-raw.tar.gz").exists()

    def test_skempi_style_keyword_args(self, tmp_path: Path):
        """SKEMPIDataset passes resource/source/destination/name/remove_archive."""
        archive_bytes_path = tmp_path / "remote" / "SKEMPI2_PDBs.tgz"
        _make_tar_gz(archive_bytes_path, {"PDBs/1A2K.pdb": b"ATOM"})
        archive_bytes = archive_bytes_path.read_bytes()

        root = tmp_path / "root"
        root.mkdir()

        def fake_download(source, destination, *, filename=None, checksum=None):
            assert source == (
                "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
            )
            assert filename == "SKEMPI-v2.0.tar.gz"
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / filename
            path.write_bytes(archive_bytes)
            return path

        with patch(
            "beignet.io._download_and_extract_archive.download",
            side_effect=fake_download,
        ):
            result = download_and_extract_archive(
                resource=(
                    "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
                ),
                source=root,
                destination=root,
                name="SKEMPI-v2.0.tar.gz",
                remove_archive=True,
            )

        assert result == root
        assert (root / "PDBs" / "1A2K.pdb").read_bytes() == b"ATOM"
        assert not (root / "SKEMPI-v2.0.tar.gz").exists()

    def test_exports_from_beignet_io(self):
        import beignet.io

        assert hasattr(beignet.io, "download_and_extract_archive")
        assert hasattr(beignet.io, "download")
        assert hasattr(beignet.io, "extract_archive")
