from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from brokk_anvil import __version__
from brokk_anvil import launcher


class ReleaseTargetTests(unittest.TestCase):
    def test_supported_release_targets(self) -> None:
        cases = [
            ("Darwin", "arm64", False, "", "universal-apple-darwin"),
            ("Darwin", "x86_64", False, "", "universal-apple-darwin"),
            ("Linux", "x86_64", False, "glibc", "x86_64-unknown-linux-gnu"),
            ("Linux", "aarch64", False, "glibc", "aarch64-unknown-linux-gnu"),
            ("Linux", "aarch64", True, "", "aarch64-linux-android"),
            ("Windows", "AMD64", False, "", "x86_64-pc-windows-msvc"),
        ]
        for system, machine, android, libc, expected in cases:
            with self.subTest(system=system, machine=machine, android=android):
                self.assertEqual(
                    launcher._release_target(
                        system=system,
                        machine=machine,
                        android=android,
                        libc=libc,
                    ),
                    expected,
                )

    def test_musl_and_unknown_architectures_are_rejected(self) -> None:
        with self.assertRaisesRegex(launcher.AnvilInstallError, "requires glibc"):
            launcher._release_target(
                system="Linux", machine="x86_64", android=False, libc="musl"
            )
        with self.assertRaisesRegex(launcher.AnvilInstallError, "riscv64"):
            launcher._release_target(
                system="Linux", machine="riscv64", android=False, libc="glibc"
            )


class CacheTests(unittest.TestCase):
    def test_cache_override_and_versioned_binary_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"BROKK_ANVIL_CACHE_DIR": directory}):
                expected = Path(directory) / __version__ / "test-target" / launcher._binary_filename()
                self.assertEqual(launcher._binary_path("test-target"), expected)

    def test_stale_download_lock_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / f"{__version__}-test-target.lock"
            lock.write_text("stale", encoding="utf-8")
            old = launcher.time.time() - launcher._LOCK_STALE_SECONDS - 1
            os.utime(lock, (old, old))
            with mock.patch.object(launcher, "_cache_root", return_value=root):
                with launcher._download_lock("test-target"):
                    self.assertTrue(lock.exists())
                    self.assertNotEqual(lock.read_text(encoding="utf-8"), "stale")
            self.assertFalse(lock.exists())


class ArchiveTests(unittest.TestCase):
    def test_checksum_sidecar_is_required_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "asset.zip"
            sidecar = root / "asset.zip.sha256"
            archive.write_bytes(b"verified bytes")
            sidecar.write_text(
                f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  asset.zip\n",
                encoding="utf-8",
            )
            launcher._verify_archive(archive, sidecar)

            sidecar.write_text(f"{'0' * 64}  asset.zip\n", encoding="utf-8")
            with self.assertRaisesRegex(launcher.AnvilInstallError, "checksum mismatch"):
                launcher._verify_archive(archive, sidecar)

            sidecar.write_text("not-a-checksum\n", encoding="utf-8")
            with self.assertRaisesRegex(launcher.AnvilInstallError, "invalid SHA-256"):
                launcher._verify_archive(archive, sidecar)

            sidecar.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(launcher.AnvilInstallError, "invalid SHA-256"):
                launcher._verify_archive(archive, sidecar)

    def test_extracts_only_the_exact_release_binary(self) -> None:
        target = "x86_64-unknown-linux-gnu"
        expected_member = f"brokk-anvil-v{__version__}-{target}/{launcher._binary_filename()}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "release.zip"
            destination = root / "cache" / launcher._binary_filename()
            payload = b"a" * launcher._MIN_BINARY_BYTES
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(expected_member, payload)
                archive.writestr("unexpected/credentials.txt", "do not extract")

            launcher._extract_binary(archive_path, destination, target)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse((root / "unexpected" / "credentials.txt").exists())

    def test_rejects_an_archive_without_the_exact_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "release.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("somewhere/anvil", b"x" * launcher._MIN_BINARY_BYTES)
            with self.assertRaisesRegex(launcher.AnvilInstallError, "does not contain expected"):
                launcher._extract_binary(
                    archive_path,
                    root / "anvil",
                    "x86_64-unknown-linux-gnu",
                )


class ResolutionTests(unittest.TestCase):
    def test_valid_cached_binary_skips_network(self) -> None:
        cached = Path("/cache/anvil")
        with (
            mock.patch.object(launcher, "_release_target", return_value="test-target"),
            mock.patch.object(launcher, "_binary_path", return_value=cached),
            mock.patch.object(launcher, "_binary_version_matches", return_value=True),
            mock.patch.object(launcher, "_download") as download,
        ):
            self.assertEqual(launcher.resolve_binary(), cached)
        download.assert_not_called()

    def test_cold_cache_downloads_archive_and_sidecar_before_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cache" / "anvil"
            downloaded: list[str] = []

            def fake_download(url: str, output: Path) -> None:
                downloaded.append(url)
                output.write_bytes(b"payload")

            def fake_extract(_archive: Path, output: Path, _target: str) -> None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("binary", encoding="utf-8")

            validity = iter([False, False, True])
            with (
                mock.patch.object(launcher, "_release_target", return_value="test-target"),
                mock.patch.object(launcher, "_binary_path", return_value=destination),
                mock.patch.object(
                    launcher, "_binary_version_matches", side_effect=lambda _path: next(validity)
                ),
                mock.patch.object(launcher, "_download_lock", return_value=contextlib.nullcontext()),
                mock.patch.object(launcher, "_download", side_effect=fake_download),
                mock.patch.object(launcher, "_verify_archive") as verify,
                mock.patch.object(launcher, "_extract_binary", side_effect=fake_extract) as extract,
            ):
                self.assertEqual(launcher.resolve_binary(), destination)

            archive = f"brokk-anvil-v{__version__}-test-target.zip"
            self.assertEqual(
                downloaded,
                [launcher._asset_url(archive), launcher._asset_url(f"{archive}.sha256")],
            )
            verify.assert_called_once()
            extract.assert_called_once()


class EntrypointTests(unittest.TestCase):
    def test_unix_entrypoint_replaces_itself_and_forwards_arguments(self) -> None:
        binary = Path("/tmp/native-anvil")
        captured: list[object] = []

        def fake_exec(path: str, arguments: list[str]) -> None:
            captured.extend([path, arguments])
            raise RuntimeError("exec called")

        with (
            mock.patch.object(launcher, "resolve_binary", return_value=binary),
            mock.patch.object(launcher.os, "execv", side_effect=fake_exec),
            mock.patch.object(launcher.os, "name", "posix"),
            mock.patch.object(sys, "argv", ["anvil", "--version"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "exec called"):
                launcher.main()

        self.assertEqual(captured, [str(binary), [str(binary), "--version"]])

    def test_install_errors_are_plain_and_nonzero(self) -> None:
        error = launcher.AnvilInstallError("unsupported platform")
        stderr = io.StringIO()
        with (
            mock.patch.object(launcher, "resolve_binary", side_effect=error),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(launcher.main(), 1)
        self.assertEqual(stderr.getvalue(), "anvil: unsupported platform\n")


if __name__ == "__main__":
    unittest.main()
