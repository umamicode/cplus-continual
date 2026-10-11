#!/usr/bin/env python3
"""Build a deterministic source-only ZIP after offline integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATTERNS = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements-tpu.txt",
    "configs/reproduction_matrix.json",
    "data/README.md",
    "experiments/**/*.md",
    "experiments/**/*.sh",
    "jax_sft/**/*.py",
    "jax_sft/**/*.sh",
    "jax_sft/**/README.md",
    "scripts/*.py",
    "tests/test_*.py",
)


def source_files() -> list[Path]:
    selected: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.name.startswith("._") or "__pycache__" in path.parts:
                continue
            if path.is_symlink() or ROOT not in path.resolve().parents:
                raise ValueError(f"source must not be a symbolic link: {path.name}")
            if path.is_file():
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite: {output}")
    if ROOT == output.parent or ROOT in output.parents:
        raise SystemExit("place the ZIP outside the source directory")
    if not output.parent.is_dir():
        raise SystemExit(f"output directory does not exist: {output.parent}")

    paths = source_files()
    with tempfile.TemporaryDirectory(prefix="cplus-source-release-") as temp:
        stage = Path(temp) / "src"
        records: list[str] = []
        for path in paths:
            relative = path.relative_to(ROOT)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            records.append(f"{digest}  ./{relative.as_posix()}\n")
        manifest = "".join(records)
        (stage / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-B", str(stage / "scripts" / "check_release.py")],
            check=True,
        )
        # Mode x prevents replacement even if another process created the file.
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if not path.is_file():
                    continue
                name = "src/" + path.relative_to(stage).as_posix()
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100644 << 16)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        # A checksum manifest is generated metadata, not an experimental output.
        (ROOT / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise SystemExit("archive CRC verification failed")
    print(f"created {output.name}: {output.stat().st_size} bytes")
    print(f"SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
