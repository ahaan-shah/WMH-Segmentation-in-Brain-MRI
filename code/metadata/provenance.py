"""Provenance capture for generated artefacts (CLAUDE.md Section 5.1).

Every generated CSV, figure, or NIfTI must record the generating script, the
git commit hash, the timestamp, and the config used, so that a stranger with
only outputs/ could regenerate every file. write_manifest() writes that
record as a JSON sidecar next to the artefact.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from metadata.config import DATASET_CONFIG, PROJECT_ROOT


def get_git_commit_hash() -> str:
    """Return the current git commit hash, or 'uncommitted' / 'unknown' if unavailable.

    Never raises — provenance capture must not crash a pipeline over a git
    lookup failing (e.g. detached state, no commits yet).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                return f"{commit}-dirty"
            return commit
    except (subprocess.SubprocessError, OSError):
        pass
    return "unknown"


def capture_provenance(generating_script: str, extra: dict | None = None) -> dict:
    """Build the provenance record: script, commit hash, timestamp, config used."""
    record = {
        "generating_script": generating_script,
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": DATASET_CONFIG,
    }
    if extra:
        record["extra"] = extra
    return record


def write_manifest(output_path: Path, generating_script: str, extra: dict | None = None) -> Path:
    """Write a <output_path>.json sidecar manifest recording this artefact's provenance."""
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    record = capture_provenance(generating_script, extra=extra)
    with open(manifest_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return manifest_path
