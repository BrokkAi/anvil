#!/usr/bin/env python3
"""Stage release notices into the Python wheel; generated files are not source."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
target = root / "crates/anvil-client-python/python/anvil_client/licenses"
target.mkdir(parents=True, exist_ok=True)
for name in ("THIRD_PARTY_LICENSES.html", "SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt", "GPL-3.0.md", "SOURCE.md"):
    shutil.copyfile(root / "licenses" / name, target / name)
print("Staged Python client license and source notices")
