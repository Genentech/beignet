import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beignet.io import check_integrity, download


class TestDownload:
    def test_download_skips_when_file_exists_without_checksum(self, tmp_path: Path):
        destination = tmp_path / "data"
        destination.mkdir()
        existing = destination / "file.bin"
        existing.write_bytes(b"cached")

        with patch("beignet.io._download._download_url") as download_url:
            path = download(
                "https://example.com/file.bin",
                destination,
                filename="file.bin",
            )

        assert path == existing
        assert path.read_bytes() == b"cached"
        download_url.assert_not_called()

    def test_download_http_writes_file(self, tmp_path: Path):
        destination = tmp_path / "data"
        payload = b"hello-archive"

        def fake_download(url: str, path: Path) -> None:
            assert url == "https://example.com/archive.tar.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        with patch("beignet.io._download._download_url", side_effect=fake_download):
            path = download(
                "https://example.com/archive.tar.gz",
                destination,
            )

        assert path == destination / "archive.tar.gz"
        assert path.read_bytes() == payload

    def test_download_verifies_md5_checksum(self, tmp_path: Path):
        destination = tmp_path / "data"
        payload = b"checksum-me"
        digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()

        def fake_download(url: str, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        with patch("beignet.io._download._download_url", side_effect=fake_download):
            path = download(
                "https://example.com/file.bin",
                destination,
                filename="file.bin",
                checksum=digest,
            )

        assert check_integrity(path, digest)

    def test_download_raises_on_checksum_mismatch(self, tmp_path: Path):
        destination = tmp_path / "data"

        def fake_download(url: str, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wrong-bytes")

        with patch("beignet.io._download._download_url", side_effect=fake_download):
            with pytest.raises(RuntimeError, match="integrity check"):
                download(
                    "https://example.com/file.bin",
                    destination,
                    filename="file.bin",
                    checksum="00000000000000000000000000000000",
                )

        assert not (destination / "file.bin").exists()

    def test_download_s3_requires_boto3(self, tmp_path: Path):
        import beignet.io._download as download_module

        with patch.object(download_module, "_download_s3") as mocked:
            # Ensure public API routes s3:// to the S3 helper.
            mocked.side_effect = ImportError(
                "Downloading from s3:// URLs requires boto3. Try: pip install boto3"
            )
            with pytest.raises(ImportError, match="boto3"):
                download(
                    "s3://bucket/key.parquet",
                    tmp_path,
                    filename="key.parquet",
                )
            mocked.assert_called_once()

    def test_download_s3_uses_helper(self, tmp_path: Path):
        destination = tmp_path / "data"
        path = destination / "object.parquet"

        def fake_s3(url: str, target: Path) -> None:
            assert url == "s3://beignet-data-dev/designdb/lake/file.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"s3-bytes")

        with patch("beignet.io._download._download_s3", side_effect=fake_s3) as mocked:
            result = download(
                "s3://beignet-data-dev/designdb/lake/file.parquet",
                destination,
                filename="object.parquet",
            )

        assert result == path
        assert path.read_bytes() == b"s3-bytes"
        mocked.assert_called_once()

    def test_download_s3_helper_with_boto3(self, tmp_path: Path):
        from beignet.io._download import _download_s3

        fake_client = MagicMock()

        def fake_download_file(bucket, key, filename):
            assert bucket == "bucket"
            assert key == "path/key.parquet"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).write_bytes(b"from-s3")

        fake_client.download_file.side_effect = fake_download_file
        fake_boto3 = MagicMock()
        fake_boto3.client.return_value = fake_client

        target = tmp_path / "key.parquet"
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            _download_s3("s3://bucket/path/key.parquet", target)

        assert target.read_bytes() == b"from-s3"
        fake_boto3.client.assert_called_once_with("s3")
