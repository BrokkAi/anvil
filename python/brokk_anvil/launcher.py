"""Resolve, verify, cache, and launch the native Anvil release binary."""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

from brokk_anvil import __version__

_RELEASE_BASE_URL = "https://github.com/BrokkAi/anvil/releases/download"
_DOWNLOAD_TIMEOUT_SECONDS = 300
_LOCK_TIMEOUT_SECONDS = 600
_LOCK_STALE_SECONDS = 1800
_VERSION_PROBE_TIMEOUT_SECONDS = 5
_MIN_BINARY_BYTES = 1024 * 1024
_MAX_BINARY_BYTES = 512 * 1024 * 1024
_CHECKSUM_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class AnvilInstallError(Exception):
    """Raised when the native Anvil binary cannot be installed safely."""


def _is_android() -> bool:
    return (
        sys.platform == "android"
        or hasattr(sys, "getandroidapilevel")
        or "ANDROID_ROOT" in os.environ
    )


def _release_target(
    *,
    system: str | None = None,
    machine: str | None = None,
    android: bool | None = None,
    libc: str | None = None,
) -> str:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    android = _is_android() if android is None else android

    if android:
        if machine in ("aarch64", "arm64"):
            return "aarch64-linux-android"
        raise AnvilInstallError(
            f"brokk-anvil {__version__} does not ship an Android binary for {machine}"
        )
    if system == "Darwin":
        if machine in ("aarch64", "arm64", "x86_64", "amd64"):
            return "universal-apple-darwin"
        raise AnvilInstallError(
            f"brokk-anvil {__version__} does not ship a macOS binary for {machine}"
        )
    if system == "Linux":
        libc_name = (libc if libc is not None else platform.libc_ver()[0]).lower()
        if libc_name and libc_name != "glibc":
            raise AnvilInstallError(
                f"brokk-anvil requires glibc on Linux; detected {libc_name}. "
                "Install Anvil with Cargo on this platform"
            )
        if machine in ("x86_64", "amd64"):
            return "x86_64-unknown-linux-gnu"
        if machine in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu"
        raise AnvilInstallError(
            f"brokk-anvil {__version__} does not ship a Linux binary for {machine}"
        )
    if system == "Windows":
        if machine in ("x86_64", "amd64"):
            return "x86_64-pc-windows-msvc"
        raise AnvilInstallError(
            f"brokk-anvil {__version__} does not ship a Windows binary for {machine}"
        )
    raise AnvilInstallError(f"unsupported platform: {system} {machine}")


def _cache_root() -> Path:
    override = os.environ.get("BROKK_ANVIL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "brokk-anvil"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "brokk-anvil"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "brokk-anvil"


def _binary_filename() -> str:
    return "anvil.exe" if os.name == "nt" else "anvil"


def _binary_path(target: str) -> Path:
    return _cache_root() / __version__ / target / _binary_filename()


def _archive_name(target: str) -> str:
    return f"brokk-anvil-v{__version__}-{target}.zip"


def _asset_url(name: str) -> str:
    return f"{_RELEASE_BASE_URL}/v{__version__}/{name}"


def _binary_version_matches(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == f"anvil {__version__}"


@contextlib.contextmanager
def _download_lock(target: str) -> Iterator[None]:
    lock_path = _cache_root() / f"{__version__}-{target}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            finally:
                os.close(descriptor)
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS
            except FileNotFoundError:
                continue
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise AnvilInstallError(f"timed out waiting for download lock {lock_path}")
            time.sleep(0.25)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"brokk-anvil/{__version__}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as error:
        raise AnvilInstallError(f"failed to download {url}: {error}") from error


def _verify_archive(archive: Path, sidecar: Path) -> None:
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as error:
        raise AnvilInstallError(f"invalid SHA-256 sidecar for {archive.name}") from error
    if not fields or not _CHECKSUM_RE.fullmatch(fields[0]):
        raise AnvilInstallError(f"invalid SHA-256 sidecar for {archive.name}")
    expected = fields[0].lower()
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise AnvilInstallError(
            f"checksum mismatch for {archive.name}: expected {expected}, got {actual}"
        )


def _extract_binary(archive_path: Path, destination: Path, target: str) -> None:
    member_name = f"brokk-anvil-v{__version__}-{target}/{_binary_filename()}"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                member = archive.getinfo(member_name)
            except KeyError as error:
                raise AnvilInstallError(
                    f"release archive does not contain expected {member_name}"
                ) from error
            if not (_MIN_BINARY_BYTES <= member.file_size <= _MAX_BINARY_BYTES):
                raise AnvilInstallError(
                    f"release archive contains an implausible Anvil binary size: {member.file_size}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.chmod(0o755)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    except zipfile.BadZipFile as error:
        raise AnvilInstallError(f"invalid release archive {archive_path.name}") from error


def resolve_binary() -> Path:
    target = _release_target()
    destination = _binary_path(target)
    if _binary_version_matches(destination):
        return destination

    with _download_lock(target):
        if _binary_version_matches(destination):
            return destination
        archive_name = _archive_name(target)
        with tempfile.TemporaryDirectory(prefix="brokk-anvil-") as directory:
            temporary = Path(directory)
            archive = temporary / archive_name
            sidecar = temporary / f"{archive_name}.sha256"
            _download(_asset_url(archive_name), archive)
            _download(_asset_url(sidecar.name), sidecar)
            _verify_archive(archive, sidecar)
            _extract_binary(archive, destination, target)
        if not _binary_version_matches(destination):
            destination.unlink(missing_ok=True)
            raise AnvilInstallError(
                f"downloaded Anvil binary does not report version {__version__}"
            )
    return destination


def main() -> int:
    try:
        binary = resolve_binary()
        arguments = [str(binary), *sys.argv[1:]]
        if os.name == "nt":
            return subprocess.call(arguments)
        os.execv(str(binary), arguments)
    except AnvilInstallError as error:
        print(f"anvil: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"anvil: failed to launch native binary: {error}", file=sys.stderr)
        return 1
    return 0
