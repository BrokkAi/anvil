#!/usr/bin/env python3
"""Fail when the Python distribution drifts from the Anvil release version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def version_from(path: Path, pattern: str, label: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise SystemExit(f"could not read {label} version from {path}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="optional release tag, for example v0.24.4")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cargo = version_from(
        root / "Cargo.toml",
        r'^version = "([^"]+)"$',
        "Cargo package",
    )
    python = version_from(
        root / "python" / "brokk_anvil" / "__init__.py",
        r'^__version__ = "([^"]+)"$',
        "Python package",
    )
    for package in ("anvil-client", "anvil-client-python", "anvil-minimizer"):
        manifest = root / "crates" / package / "Cargo.toml"
        version = version_from(manifest, r'^version = "([^"]+)"$', package)
        if version != cargo:
            raise SystemExit(f"{package} version {version} differs from Anvil {cargo}")
    binding = (root / "crates/anvil-client-python/Cargo.toml").read_text()
    if f'version = "{cargo}"' not in next(line for line in binding.splitlines() if line.startswith("anvil-client =")):
        raise SystemExit("Python binding dependency differs from Anvil version")
    expected = args.tag.removeprefix("v") if args.tag else cargo
    if cargo != python or python != expected:
        raise SystemExit(
            "version lockstep violated: "
            f"Cargo.toml={cargo}, python={python}, expected={expected}"
        )
    print(f"Python package version is in lockstep: {python}")


if __name__ == "__main__":
    main()
