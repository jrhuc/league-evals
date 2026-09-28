import hashlib
import json
from pathlib import Path

from league_evals.provenance import engine_provenance


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


def test_checked_in_engine_patch_matches_lock():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "engine.lock.json").read_text())
    assert len(lock["commit"]) == 40
    assert (
        hashlib.sha256((root / lock["patch"]["path"]).read_bytes()).hexdigest()
        == lock["patch"]["sha256"]
    )
