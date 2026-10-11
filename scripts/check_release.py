#!/usr/bin/env python3
"""Check an unpacked source release without importing accelerator libraries."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def check_manifest(root: Path) -> int:
    listed: set[str] = set()
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.removeprefix("./")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in listed:
            raise ValueError(f"unsafe or duplicate manifest entry: {name}")
        listed.add(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symbolic-link source: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and path.relative_to(root).parts[0] != ".git"
    }
    if actual != listed:
        raise ValueError(f"files missing from manifest: {sorted(actual - listed)}")
    return len(listed)


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "audit_anonymity.py"),
            "--allow-public-authorship",
            str(ROOT),
        ],
        check=True,
    )
    count = check_manifest(ROOT)
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted(ROOT.rglob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"release checks passed: {count} checksums; {len(python_files)} Python files")
    print("Static checks only; no accelerator execution or experiment was performed.")


if __name__ == "__main__":
    main()
