#!/usr/bin/env python3
"""Fail if a source release contains credentials or machine-private material."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".netrc",
    "credentials.json",
    "service-account.json",
}
FORBIDDEN_DIRECTORIES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "__MACOSX",
}
PRIVATE_PATTERNS = {
    "macOS user path": re.compile(r"/Users/[A-Za-z0-9._-]+"),
    "Linux home path": re.compile(r"/home/[A-Za-z0-9._-]+"),
    "Windows user path": re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+"),
    "root home path": re.compile(r"/root/[A-Za-z0-9._/-]+"),
    "scratch path": re.compile(r"/scratch/[A-Za-z0-9._/-]+"),
    "mounted workspace path": re.compile(r"/(?:mnt|workspace)/[A-Za-z0-9._/-]+"),
    "cloud bucket": re.compile(r"(?:gs|s3)://[^\s'\"]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "authenticated URL": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "provider credential": re.compile(
        r"(?:AKIA|ASIA)[0-9A-Z]{16}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|hf_[A-Za-z0-9]{20,}"
        r"|sk-[A-Za-z0-9_-]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    ),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
        r"client[_-]?secret|private[_-]?key)\s*[:=]\s*['\"][^'\"${}<]{8,}['\"]"
    ),
    "account identifier": re.compile(
        r"(?i)(?:wandb[_-]?entity|hf[_-]?username|account[_-]?id|"
        r"gcp[_-]?project|project[_-]?id)\s*[:=]\s*['\"][A-Za-z0-9._-]+['\"]"
    ),
    "author metadata": re.compile(r"(?im)^\s*(?:__author__\s*=|author\s*:|created by\s+)"),
    "named copyright notice": re.compile(r"(?i)copyright\s+(?:\(c\)\s*)?20\d{2}\s+[A-Za-z]"),
}


def local_identity_patterns(root: Path) -> dict[str, re.Pattern[str]]:
    """Build checks from local account and Git identity without recording them."""
    candidates = {
        getpass.getuser(),
        Path.home().name,
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    }
    for key in ("user.name", "user.email"):
        result = subprocess.run(
            ["git", "-C", str(root), "config", "--get", key],
            check=False,
            capture_output=True,
            text=True,
        )
        candidates.add(result.stdout.strip())
    ignored = {"", "root", "user", "ubuntu", "runner", "unknown"}
    identities = sorted(value for value in candidates if len(value) >= 4 and value.lower() not in ignored)
    return {
        f"local author identity {index}": re.compile(re.escape(value), re.IGNORECASE)
        for index, value in enumerate(identities)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument(
        "--allow-public-authorship",
        action="store_true",
        help="permit author names in public metadata such as a citation block",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings: list[str] = []
    self_path = Path(__file__).resolve()
    patterns = dict(PRIVATE_PATTERNS)
    if not args.allow_public_authorship:
        patterns.update(local_identity_patterns(args.root))
    for path in sorted(args.root.rglob("*")):
        relative = path.relative_to(args.root)
        # A working tree's own Git metadata is not part of the published
        # source. Nested repositories are still rejected below.
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            findings.append(f"{path}: symbolic link")
            continue
        if any(part in FORBIDDEN_DIRECTORIES or part.startswith(".venv") for part in relative.parts):
            findings.append(f"{path}: forbidden directory")
            continue
        if path.is_file() and (path.name in FORBIDDEN_NAMES or path.name.startswith("._")):
            findings.append(f"{path}: forbidden private file")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == self_path:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path}:{line}: {label}")
    if findings:
        raise SystemExit("privacy audit failed:\n" + "\n".join(findings))
    print(f"privacy audit passed: {args.root.resolve()}")


if __name__ == "__main__":
    main()
