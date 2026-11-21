"""Create release decisions and immutable release manifests."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

RELEASE_DIRECTORY: Path = Path("artifacts/releases")


def git_commit() -> str:
    """This function returns the current Git commit hash."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    """This function computes a stable SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
