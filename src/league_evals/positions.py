"""Recorded decisions selected with independent choice and scoring rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log

from .bridge import BridgeError, request_sync
from .bridge import league_dir as engine_dir
from .provenance import engine_provenance

DATA = Path(__file__).resolve().parent / "data" / "positions"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "data" / "positions"


def position_path(name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("positions must be a name, not a path")
    return DATA / f"{name}.json"


def load_positions(name: str) -> dict[str, Any]:
    path = position_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `league-positions build RUN_DIR --name {name}`"
        )
    return json.loads(path.read_text())


def read_run(run_dir: Path) -> list[dict[str, Any]]:
    games = []
    with sqlite3.connect(
        (run_dir / "league.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        rows = db.execute(
            "SELECT s.series_id, s.identity_json, g.game_number, g.attempt_id, "
            "g.seed_json, g.log_path FROM series s JOIN series_games g "
            "ON s.series_id = g.series_id ORDER BY s.series_index, g.game_number"
        ).fetchall()
    decisions = {}
    for series, identity_json, number, attempt, seed, log_path in rows:
        identity = json.loads(identity_json)
        choices = {}
        for pid in ("p1", "p2"):
            key = (series, pid)
            if key not in decisions:
                path = run_dir / "series" / series / f"{pid}-decisions.jsonl"
                decisions[key] = [
                    json.loads(line) for line in path.read_text().splitlines() if line
                ]
            choices[pid] = [
                row["action"]
                for row in decisions[key]
                if row.get("kind") == "decision"
                and row.get("game_number") == number
                and row.get("attempt_id") == attempt
                and row.get("outcome") == "accepted"
            ]
        games.append(
            {
                "id": f"{series}-{number}",
                "source": {
                    "format": identity["format"],
                    "seed": json.loads(seed),
                    "names": {pid: f"{pid}-{model}" for pid, model in identity["players"].items()},
                    "packed": identity["packed_teams"],
                    "choices": choices,
                },
                "log": (run_dir / log_path).read_text().split("\n"),
                "models": identity["players"],
            }
        )
    return games


def _value_shard(inputs: list[dict], directory: Path) -> list[dict]:
    rows = []
    with tempfile.TemporaryFile(mode="w+") as source:
        for row in inputs:
            source.write(json.dumps(row) + "\n")
        source.seek(0)
        with subprocess.Popen(
            ["env", "-u", "NODE_OPTIONS", "node", str(directory / "dist/src/cli.js"), "positions"],
            cwd=directory,
            stdin=source,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                for line in process.stdout:
                    row = json.loads(line)
                    if row.get("kind") not in {"game", "position"}:
                        raise BridgeError(f"unexpected positions output: {row}")
                    rows.append(row)
                    detail = row.get("choice_index", row.get("verified"))
                    print(f"positions: {row['id']} {row['kind']} {detail}", file=sys.stderr)
                if process.wait():
                    raise BridgeError(f"positions process exited with {process.returncode}")
            except BaseException:
                process.kill()
                raise
    if sorted(r["id"] for r in rows if r["kind"] == "game") != sorted(r["id"] for r in inputs):
        raise BridgeError("positions returned an incomplete game sequence")
    return rows


def value_games(
    games: list[dict],
    settings: dict,
    jobs: int,
    only: dict | None = None,
    cache: Path | None = None,
    *,
    league_dir: Path | None = None,
) -> list[dict]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    inputs = [
        {
            **{k: game[k] for k in ("id", "source", "log")},
            "settings": settings,
            **({"only": only[game["id"]]} if only is not None else {}),
        }
        for game in games
        if only is None or game["id"] in only
    ]
    if not inputs:
        return []
    directory = league_dir or engine_dir()
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    path = None
    if cache is not None:
        path = (
            Path(cache)
            / f"pass-{settings['samples']}-{settings['salt']}{'-only' if only is not None else ''}.jsonl"
        )
        if path.exists():
            cached = [json.loads(line) for line in path.read_text().splitlines()]
            if cached and cached[0] == {"kind": "cache", "sha256": signature}:
                rows = cached[1:]
                if sorted(r["id"] for r in rows if r["kind"] == "game") == sorted(
                    g["id"] for g in inputs
                ):
                    print(f"positions: reusing {path}", file=sys.stderr)
                    return rows
    count = min(jobs, len(inputs))
    print(f"positions: {len(inputs)} games, {count} workers, settings={settings}", file=sys.stderr)
    rows = []
    with ThreadPoolExecutor(max_workers=count) as workers:
        pending = {workers.submit(_value_shard, inputs[i::count], directory) for i in range(count)}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                print(f"positions: {len(pending)} workers still valuing decisions", file=sys.stderr)
            for future in done:
                rows.extend(future.result())
    rows.sort(key=lambda r: (r["id"], r["kind"], r.get("focal", ""), r.get("choice_index", -1)))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            output.write(json.dumps({"kind": "cache", "sha256": signature}) + "\n")
            for row in rows:
                output.write(json.dumps(row) + "\n")
        os.replace(output.name, path)
    return rows


def decision_key(row: dict) -> tuple[str, str, int]:
    return row["id"], row["focal"], row["choice_index"]


def shortlists(rows: list[dict], *, shortlist: int = 12) -> dict:
    only = defaultdict(list)
    for row in rows:
        if row["kind"] != "position":
            continue
        commands = {a["command"] for a in row["actions"]}
        if row["greedy"] not in commands or len(commands) < 2:
            continue
        ranked = sorted(row["actions"], key=lambda a: (-a["value"], a["command"]))
        selected = {a["command"] for a in ranked[:shortlist]} | {row["greedy"]}
        if row["recorded"] in commands:
            selected.add(row["recorded"])
        only[row["id"]].append(
            {"pid": row["focal"], "choice_index": row["choice_index"], "commands": sorted(selected)}
        )
    return dict(only)


def regret(position: dict, command: str, entry: dict | None = None) -> dict | None:
    entry = entry if entry is not None else position["values"].get(command)
    if entry is None and command == "forfeit":
        entry = {"value": 0.0, "explored": 0.0, "label": "Forfeit"}
    if entry is None:
        return None
    value = entry["value"]
    best = position["values"][position["best"]]["value"]
    return {
        "value": value,
        "regret": best - value,
        "good": float(
            command in position["good"] or value >= best - position.get("good_margin", 0.10)
        ),
        "beats_greedy": float(value > position["values"][position["greedy"]]["value"]),
        "matches_recorded": float(command == position["recorded"]["command"]),
    }


def select_positions(
    games: list[dict],
    coarse: list[dict],
    choose: list[dict],
    select: list[dict],
    score: list[dict],
    *,
    gap: float = 0.20,
    explored_gap: float = 0.10,
    good_margin: float = 0.10,
) -> list[dict]:
    sources = {game["id"]: game for game in games}
    a = {decision_key(r): r for r in choose if r["kind"] == "position"}
    b = {decision_key(r): r for r in select if r["kind"] == "position"}
    c = {decision_key(r): r for r in score if r["kind"] == "position"}
    selected = []
    for row in coarse:
        if row["kind"] != "position" or len(row["actions"]) < 2:
            continue
        if row["greedy"] not in {v["command"] for v in row["actions"]}:
            continue
        key = decision_key(row)
        if key not in a or key not in b or key not in c or not a[key]["actions"]:
            continue
        ranked = sorted(a[key]["actions"], key=lambda v: (-v["value"], v["command"]))
        best = ranked[0]
        gate = {v["command"]: v for v in b[key]["actions"]}
        values = {
            v["command"]: {k: v[k] for k in ("value", "explored", "label")}
            for v in c[key]["actions"]
        }
        if any(
            name not in table
            for name in (best["command"], row["greedy"])
            for table in (gate, values)
        ):
            continue
        chosen, greedy = gate[best["command"]], gate[row["greedy"]]
        if (
            chosen["value"] - greedy["value"] < gap
            or chosen["explored"] - greedy["explored"] < explored_gap
        ):
            continue
        game, focal, index = key
        models = sources[game]["models"]
        position = {
            "id": f"{game}-{focal}-{index}",
            "game": game,
            "focal": focal,
            "turn": row["turn"],
            "choice_index": index,
            "opponent_choice_index": row["opponent_choice_index"],
            "recorded": {
                "model": models[focal],
                "opponent_model": models["p2" if focal == "p1" else "p1"],
                "command": row["recorded"],
            },
            "greedy": row["greedy"],
            "best": best["command"],
            "good": sorted(
                v["command"] for v in ranked if v["value"] >= best["value"] - good_margin
            ),
            "good_margin": good_margin,
            "hidden_opponents": row["hidden_opponents"],
            "actions": len(row["actions"]),
            "values": values,
        }
        position["uniform"] = {
            k: sum(regret(position, v["command"], v)[k] for v in row["actions"])
            / len(row["actions"])
            for k in ("value", "regret")
        }
        selected.append(position)
    return sorted(selected, key=lambda p: p["id"])


def build_dataset(
    run_dir: Path,
    name: str,
    *,
    jobs: int = 1,
    cache: Path | None = None,
    coarse_samples: int = 4,
    fine_samples: int = 24,
    shortlist: int = 12,
    gap: float = 0.20,
    explored_gap: float = 0.10,
    good_margin: float = 0.10,
    epsilon: float = 0.25,
    max_turns: int = 40,
) -> dict:
    if min(coarse_samples, fine_samples, shortlist, max_turns, jobs) < 1:
        raise ValueError("samples, shortlist, max_turns, and jobs must be positive")
    if any(not 0 <= v <= 1 for v in (good_margin, epsilon)):
        raise ValueError("good_margin and epsilon must be between zero and one")
    if any(not -1 <= v <= 1 for v in (gap, explored_gap)):
        raise ValueError("gap and explored_gap must be between minus one and one")
    games = read_run(run_dir)
    formats = {g["source"]["format"] for g in games}
    if len(formats) != 1:
        raise ValueError("run must contain games in exactly one format")
    format = formats.pop()
    (hello,) = request_sync([("open", {"format": format})])
    provenance = {**hello, **engine_provenance(engine_dir())}
    fine = {"samples": fine_samples, "epsilon": epsilon, "maxTurns": max_turns}
    coarse = value_games(games, {**fine, "samples": coarse_samples, "salt": 1}, jobs, cache=cache)
    only = shortlists(coarse, shortlist=shortlist)
    choose, select, score = (
        value_games(games, {**fine, "salt": salt}, jobs, only=only, cache=cache)
        for salt in (11, 12, 13)
    )
    unverified = sorted(
        {
            r["id"]
            for r in coarse + choose + select + score
            if r["kind"] == "game" and not r["verified"]
        }
    )
    selected = select_positions(
        games,
        coarse,
        choose,
        select,
        score,
        gap=gap,
        explored_gap=explored_gap,
        good_margin=good_margin,
    )
    selected = [p for p in selected if p["game"] not in unverified]
    kept = {p["game"] for p in selected}
    print(f"positions: selected {len(selected)}; unverified games: {unverified}", file=sys.stderr)
    return {
        "id": name,
        "format": format,
        "run": run_dir.name,
        "selection": {
            "coarse": {"samples": coarse_samples, "salt": 1},
            "fine": fine,
            "salts": {"choose": 11, "select": 12, "score": 13},
            "shortlist": shortlist,
            "gap": gap,
            "explored_gap": explored_gap,
            "good_margin": good_margin,
            "games": len(games),
            "unverified_games": unverified,
            "decisions_valued": sum(r["kind"] == "position" for r in coarse),
            "selected": len(selected),
            "uniform": "coarse estimate over all legal actions",
        },
        "provenance": provenance,
        "games": {
            g["id"]: {k: g[k] for k in ("source", "log", "models")}
            for g in games
            if g["id"] in kept
        },
        "positions": selected,
    }


def value_command(
    dataset: dict, position: dict, command: str, league_dir: Path | None = None
) -> dict | None:
    game = {"id": position["game"], **dataset["games"][position["game"]]}
    settings = {**dataset["selection"]["fine"], "salt": dataset["selection"]["salts"]["score"]}
    only = {
        game["id"]: [
            {
                "pid": position["focal"],
                "choice_index": position["choice_index"],
                "commands": [command],
            }
        ]
    }
    rows = value_games([game], settings, 1, only=only, league_dir=league_dir)
    for row in rows:
        if row["kind"] == "position" and decision_key(row) == (
            game["id"],
            position["focal"],
            position["choice_index"],
        ):
            for action in row["actions"]:
                if action["command"] == command:
                    return {k: action[k] for k in ("value", "explored", "label")}
    return None


def verify_dataset(dataset: dict, jobs: int = 1) -> dict:
    settings = {**dataset["selection"]["fine"], "salt": dataset["selection"]["salts"]["score"]}
    only = defaultdict(list)
    for position in dataset["positions"]:
        only[position["game"]].append(
            {
                "pid": position["focal"],
                "choice_index": position["choice_index"],
                "commands": list(position["values"]),
            }
        )
    games = [{"id": name, **game} for name, game in dataset["games"].items()]
    rows = value_games(games, settings, jobs, only=dict(only))
    recomputed = {
        decision_key(row): {action["command"]: action for action in row["actions"]}
        for row in rows
        if row["kind"] == "position"
    }
    values = 0
    mismatches = []
    for position in dataset["positions"]:
        table = recomputed.get((position["game"], position["focal"], position["choice_index"]), {})
        for command, stored in position["values"].items():
            values += 1
            fresh = table.get(command)
            if fresh is None or any(fresh[k] != stored[k] for k in ("value", "explored")):
                mismatches.append(
                    {
                        "position": position["id"],
                        "command": command,
                        "stored": {k: stored[k] for k in ("value", "explored")},
                        "recomputed": fresh and {k: fresh[k] for k in ("value", "explored")},
                    }
                )
    return {"positions": len(dataset["positions"]), "values": values, "mismatches": mismatches}


def restamp(dataset: dict, values: int) -> dict:
    (hello,) = request_sync([("open", {"format": dataset["format"]})])
    return {
        **dataset,
        "provenance": {
            **hello,
            **engine_provenance(engine_dir()),
            "revalued": {"from": dataset["provenance"]["harness_commit"], "values": values},
        },
    }


def summary(rows: list[dict | None]) -> dict:
    scored = [r for r in rows if r is not None]
    return {
        "n": len(rows),
        "scored": len(scored),
        **{
            name: sum(r[key] for r in scored) / len(scored)
            if scored and all(key in r for r in scored)
            else None
            for name, key in (
                ("mean_regret", "regret"),
                ("good_rate", "good"),
                ("beats_greedy_rate", "beats_greedy"),
            )
        },
    }


def baselines(dataset: dict) -> dict:
    positions = dataset["positions"]
    models = sorted({p["recorded"]["model"] for p in positions})
    return {
        "greedy": summary([regret(p, p["greedy"]) for p in positions]),
        "recorded": summary([regret(p, p["recorded"]["command"]) for p in positions]),
        "recorded_models": {
            m: summary(
                [
                    regret(p, p["recorded"]["command"])
                    for p in positions
                    if p["recorded"]["model"] == m
                ]
            )
            for m in models
        },
        "uniform": {
            **summary([p.get("uniform") for p in positions]),
            "valuation": "coarse estimate over all legal actions",
        },
    }


def report(log_dir: Path) -> dict:
    groups = defaultdict(list)
    datasets = {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        if log.eval.task.split("/")[-1] != "vgc_position":
            continue
        for sample in log.samples or []:
            meta, data = sample.metadata or {}, sample.store or {}
            name = meta["positions"]
            if name not in datasets:
                datasets[name] = load_positions(name)
            score = (sample.scores or {}).get("position_regret")
            scored = bool(
                score and score.value["scored"] and not sample.error and not sample.invalidation
            )
            position = meta["position"]
            recorded = regret(position, position["recorded"]["command"])
            difference = score.value["value"] - recorded["value"] if scored and recorded else None
            groups[(log.eval.model, meta["tool_access"], name)].append(
                {
                    "log": info.name,
                    "sample_id": sample.id,
                    "epoch": sample.epoch,
                    "error": bool(sample.error),
                    "defaulted": any(d["defaulted"] for d in data.get("decisions", [])),
                    "score": score.value if scored else None,
                    "paired": None
                    if difference is None
                    else "better"
                    if difference > 0
                    else "worse"
                    if difference < 0
                    else "same",
                }
            )
    return {
        "groups": [
            {
                "model": model,
                "tool_access": access,
                "positions": name,
                **summary([r["score"] for r in rows]),
                "samples": len(rows),
                "errors": sum(r["error"] for r in rows),
                "defaulted": sum(r["defaulted"] for r in rows),
                "paired": {
                    k: sum(r["paired"] == k for r in rows) for k in ("better", "same", "worse")
                },
                "rows": rows,
            }
            for (model, access, name), rows in sorted(groups.items())
        ],
        "baselines": {name: baselines(dataset) for name, dataset in datasets.items()},
    }


def baseline_table(rows: dict) -> str:
    from .report import fmt

    lines = [
        "| baseline | n | scored | mean regret | good rate | beats greedy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    flattened = {k: v for k, v in rows.items() if k != "recorded_models"}
    flattened.update({f"recorded: {k}": v for k, v in rows["recorded_models"].items()})
    for name, row in flattened.items():
        label = name + (" (coarse estimate)" if name == "uniform" else "")
        lines.append(
            f"| {label} | {row['n']} | {row['scored']} | {fmt(row['mean_regret'])} | {fmt(row['good_rate'], True)} | {fmt(row['beats_greedy_rate'], True)} |"
        )
    return "\n".join(lines)


def report_table(result: dict) -> str:
    from .report import fmt

    lines = [
        "| model | tools | dataset | samples | scored | errors | defaulted | mean regret | good rate | beats greedy | better / same / worse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["groups"]:
        pairs = " / ".join(str(r["paired"][k]) for k in ("better", "same", "worse"))
        lines.append(
            f"| {r['model']} | {r['tool_access']} | {r['positions']} | {r['samples']} | {r['scored']} | {r['errors']} | {r['defaulted']} | {fmt(r['mean_regret'])} | {fmt(r['good_rate'], True)} | {fmt(r['beats_greedy_rate'], True)} | {pairs} |"
        )
    for name, rows in result["baselines"].items():
        lines.extend(["", name, baseline_table(rows)])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("run_dir", type=Path)
    build.add_argument("--name", required=True)
    build.add_argument("--jobs", type=int, default=1)
    build.add_argument("--cache", type=Path)
    build.add_argument("--coarse-samples", type=int, default=4)
    build.add_argument("--fine-samples", type=int, default=24)
    build.add_argument("--gap", type=float, default=0.20)
    build.add_argument("--explored-gap", type=float, default=0.10)
    baseline = commands.add_parser("baselines")
    baseline.add_argument("name")
    baseline.add_argument("--json", action="store_true")
    reports = commands.add_parser("report")
    reports.add_argument("log_dir", type=Path)
    reports.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("name")
    verify.add_argument("--jobs", type=int, default=1)
    verify.add_argument("--restamp", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            path = position_path(args.name)
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            dataset = build_dataset(
                args.run_dir,
                args.name,
                jobs=args.jobs,
                cache=args.cache,
                coarse_samples=args.coarse_samples,
                fine_samples=args.fine_samples,
                gap=args.gap,
                explored_gap=args.explored_gap,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as output:
                output.write(json.dumps(dataset, indent=1) + "\n")
            print(path)
        elif args.command == "verify":
            dataset = load_positions(args.name)
            result = verify_dataset(dataset, args.jobs)
            mismatches = result["mismatches"]
            for row in mismatches[:20]:
                print(json.dumps(row))
            print(
                f"positions: {result['positions']}; values compared: {result['values']}; "
                f"mismatches: {len(mismatches)}"
            )
            if mismatches:
                raise SystemExit(1)
            if args.restamp:
                path = position_path(args.name)
                path.write_text(json.dumps(restamp(dataset, result["values"]), indent=1) + "\n")
                print(f"restamped {path}")
        else:
            result = (
                baselines(load_positions(args.name))
                if args.command == "baselines"
                else report(args.log_dir)
            )
            print(
                json.dumps(result, indent=2, allow_nan=False)
                if args.json
                else baseline_table(result)
                if args.command == "baselines"
                else report_table(result)
            )
    except (ValueError, OSError, BridgeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
