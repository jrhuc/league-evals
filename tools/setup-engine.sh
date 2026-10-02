#!/usr/bin/env sh
# Clone the league harness at the pinned commit and build it, so `vgcleague bridge` exists.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
repo=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$here/engine.lock.json")
commit=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["commit"])' "$here/engine.lock.json")
case "$commit" in
  *[!0-9a-f]*|"") echo "engine.lock.json must pin a full commit hash (found '$commit')" >&2; exit 1 ;;
esac
if [ "${#commit}" -ne 40 ]; then echo "engine commit must be a full SHA" >&2; exit 1; fi
target="$here/engine/ai-draft-league"
if [ ! -d "$target/.git" ]; then git clone "$repo" "$target"; fi
# Re-running is safe, but never overwrite an operator's engine changes.
if [ "$(git -C "$target" rev-parse HEAD)" != "$commit" ]; then
  if [ -n "$(git -C "$target" status --porcelain)" ]; then
    echo "engine checkout has changes; use a fresh engine directory" >&2; exit 1
  fi
  git -C "$target" fetch --quiet origin "$commit"
  git -C "$target" checkout --quiet --detach "$commit"
fi
cd "$target"
pnpm install --frozen-lockfile
pnpm --dir packages/league run setup:showdown
pnpm --dir packages/league run build
python3 - "$target/packages/league" "$here/engine.lock.json" "$here/src/league_evals/provenance.py" <<'PY'
import sys, runpy
from pathlib import Path
runpy.run_path(sys.argv[3])['write_build_manifest'](Path(sys.argv[1]), Path(sys.argv[2]))
PY
echo "league harness built at $target/packages/league"
