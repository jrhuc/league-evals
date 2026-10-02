import json
import subprocess
from pathlib import Path

import pytest

from league_evals.provenance import engine_provenance, write_build_manifest


def test_build_manifest_detects_source_and_runtime_drift(tmp_path):
    files = {
        "src/cli.ts": "source",
        "dist/src/cli.js": "runtime",
        "pokemon-showdown/dist/sim/index.js": "simulator",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    original = engine_provenance(tmp_path)
    assert not original["build_verified"]
    (tmp_path / "eval-build.json").write_text(json.dumps(original))
    assert engine_provenance(tmp_path)["build_verified"]
    (tmp_path / "dist/src/cli.js").write_text("different build from same Git HEAD")
    changed = engine_provenance(tmp_path)
    assert not changed["build_verified"]
    assert changed["source_sha256"] == original["source_sha256"]
    assert changed["runtime_sha256"] != original["runtime_sha256"]


def test_engine_lock_pins_a_commit_and_its_source():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "engine.lock.json").read_text())
    assert set(lock) == {"repository", "commit", "source_sha256"}
    assert len(lock["commit"]) == 40 and len(lock["source_sha256"]) == 64


def test_build_manifest_requires_the_locked_commit_and_source(tmp_path):
    engine = tmp_path / "packages/league"
    for name in ("src/cli.ts", "dist/src/cli.js", "pokemon-showdown/dist/sim/index.js"):
        path = engine / name
        path.parent.mkdir(parents=True)
        path.write_text(name)
    for command in (["init", "-q"], ["add", "."], ["commit", "-qm", "engine"]):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
            + command,
            cwd=tmp_path,
            check=True,
        )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    source = engine_provenance(engine)["source_sha256"]
    lock = tmp_path / "engine.lock.json"
    for commit, digest in (("0" * 40, source), (head, "0" * 64)):
        lock.write_text(json.dumps({"commit": commit, "source_sha256": digest}))
        with pytest.raises(ValueError, match="engine.lock.json"):
            write_build_manifest(engine, lock)
    lock.write_text(json.dumps({"commit": head, "source_sha256": source}))
    write_build_manifest(engine, lock)
    assert engine_provenance(engine)["build_verified"]
    assert set(json.loads((engine / "eval-build.json").read_text())) == {
        "source_sha256",
        "runtime_sha256",
        "showdown_runtime_sha256",
    }
