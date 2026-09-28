"""Export a harness team pool as the packed teams a dataset needs: python -m league_evals.pools test"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .bridge import FORMAT, request_sync

DATA = Path(__file__).resolve().parent / "data" / "pools"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "data" / "pools"


def export_pool(name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("pool must be a name, not a path")
    hello, pool = request_sync([("open", {"format": FORMAT}), ("pool", {"name": name})])
    pool["provenance"] = {
        "harness_commit": hello["harness_commit"],
        "showdown_commit": hello["showdown_commit"],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(pool, indent=2) + "\n")
    return path


if __name__ == "__main__":
    for pool_name in sys.argv[1:] or ["test"]:
        print(export_pool(pool_name))
