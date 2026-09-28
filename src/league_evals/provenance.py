"""Content fingerprints: a Git HEAD alone does not identify a locally modified engine."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def tree_hash(root: Path, patterns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    if not paths:
        raise FileNotFoundError(f"No engine files found at {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def engine_provenance(directory: Path) -> dict[str, Any]:
    source = tree_hash(
        directory,
        ("src/**/*.ts", "tools/*.mjs", "package.json", "tsconfig.json", "showdown.lock.json"),
    )
    runtime = tree_hash(directory, ("dist/**/*.js",))
    showdown = directory / "pokemon-showdown"
    simulator = tree_hash(showdown, ("dist/**/*.js",))
    manifest_path = directory / "eval-build.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    hashes = {
        "source_sha256": source,
        "runtime_sha256": runtime,
        "showdown_runtime_sha256": simulator,
    }
    return {
        **hashes,
        "build_verified": all(manifest.get(k) == v for k, v in hashes.items()),
        "engine_patch_sha256": manifest.get("patch_sha256"),
    }


def write_build_manifest(directory: Path, lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text())
    expected = lock.get("source_sha256")
    manifest_path = directory / "eval-build.json"
    provenance = engine_provenance(directory)
    if expected and provenance["source_sha256"] != expected:
        raise ValueError("engine source does not match engine.lock.json")
    base = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    if base != lock["commit"]:
        raise ValueError("engine base commit does not match engine.lock.json")
    manifest_path.write_text(
        json.dumps(
            {
                **{k: v for k, v in provenance.items() if k.endswith("sha256")},
                "patch_sha256": lock["patch"]["sha256"],
            },
            indent=2,
        )
        + "\n"
    )
