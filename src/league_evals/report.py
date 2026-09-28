"""Audit retained Inspect attempts, separated by configuration: python -m league_evals.report logs/"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log

from .analysis import finite, pair_mean, paired_contrast
from .scorers import discipline_summary, mechanics_summary, outcome_summary


def parse_prices(entries: list[str]) -> dict[str, tuple[float, ...]]:
    prices = {}
    for entry in entries:
        model, sep, amounts = entry.partition("=")
        try:
            values = tuple(float(x) for x in amounts.split(","))
        except ValueError as error:
            raise ValueError("prices must be MODEL=IN,OUT[,CACHE_READ,CACHE_WRITE]") from error
        if (
            not sep
            or not model
            or len(values) not in {2, 4}
            or any(not finite(v) or v < 0 for v in values)
        ):
            raise ValueError("prices must be MODEL=IN,OUT[,CACHE_READ,CACHE_WRITE], nonnegative")
        prices[model] = values
    return prices


def sample_cost(usage: dict[str, Any], prices: dict[str, tuple[float, ...]]) -> float | None:
    if not usage:
        return None
    total = 0.0
    for model, use in usage.items():
        if use.total_cost is not None:
            total += use.total_cost
            continue
        rates = prices.get(model)
        if rates is None:
            return None
        reads, writes = use.input_tokens_cache_read or 0, use.input_tokens_cache_write or 0
        if (reads or writes) and len(rates) != 4:
            return None  # Cache discounts and write premiums are provider-specific.
        total += (use.input_tokens * rates[0] + use.output_tokens * rates[1]) / 1e6
        if len(rates) == 4:
            total += (reads * rates[2] + writes * rates[3]) / 1e6
    return total


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def configuration(log: Any, sample: Any) -> dict[str, Any]:
    args = log.eval.task_args
    meta = sample.metadata or {}
    data = sample.store or {}
    provenance = data.get("provenance", {})
    system = data.get("system_prompt") or next(
        (m.text for m in sample.messages if m.role == "system"), None
    )
    return {
        "model": log.eval.model,
        "task": log.eval.task,
        "task_version": log.eval.task_version,
        "task_sha256": data.get("task_sha256", "legacy-unrecorded"),
        "pool": args.get("pool", "test"),
        "pool_sha256": meta.get("pool_sha256", "legacy-unrecorded"),
        "format": meta.get("format"),
        "opponent": meta.get("opponent_policy", args.get("opponent", "unknown")),
        "sheets": args.get("sheets", "open"),
        "tool_access": meta.get("tool_access", args.get("tool_access", "full")),
        "league_prompt": args.get("league_prompt", False),
        "max_generations": args.get("max_generations", "unrecorded"),
        "max_decisions": args.get("max_decisions", "unrecorded"),
        "focal_seat": args.get("focal_seat", "p1"),
        "system_sha256": fingerprint(system) if system else "unrecorded",
        "tools_sha256": fingerprint(data.get("tool_catalog")),
        "engine": provenance,
        "generation": log.eval.model_generate_config.model_dump(exclude_none=True),
        "model_args": log.eval.model_args,
        "model_base_url": log.eval.model_base_url,
        "limits": {
            k: getattr(log.eval.config, k, None)
            for k in (
                "token_limit",
                "token_limit_type",
                "message_limit",
                "time_limit",
                "working_limit",
                "cost_limit",
                "retry_on_error",
            )
        },
    }


def collect(log_dir: Path, prices: dict[str, tuple[float, ...]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        for sample in log.samples or []:
            config = configuration(log, sample)
            group = fingerprint(config)[:12]
            data, meta = sample.store or {}, sample.metadata or {}
            result = data.get("outcome")
            complete = (
                bool(result)
                and not sample.error
                and not sample.invalidation
                and (data.get("completion", "complete") == "complete")
                and not any(line.startswith("|error|") for line in data.get("log", []))
            )
            row = {
                "group": group,
                "config": config,
                "log": info.name,
                "log_status": log.status,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "pair": meta.get("pair"),
                "seed": meta.get("seed"),
                "focal_team": meta.get("focal_team"),
                "opponent_team": meta.get("opponent_team"),
                "focal_seat": meta.get("focal_seat", "p1"),
                "complete": complete,
                "error": str(sample.error) if sample.error else None,
                "invalidated": bool(sample.invalidation),
                "limit": str(sample.limit) if sample.limit else None,
                "completion": data.get("completion", "legacy"),
                "retried": bool(sample.error_retries),
                "cost": sample_cost(sample.model_usage or {}, prices),
            }
            if complete:
                decisions = data.get("decisions", [])
                row.update(
                    outcome_summary(
                        result, row["focal_seat"], any(d["defaulted"] for d in decisions)
                    )
                )
                row.update(discipline_summary(decisions))
                if data.get("audit"):
                    row.update(mechanics_summary(data["audit"]))
                # Report tokens using usage, which includes cache reads and writes.
                tokens = sum(u.total_tokens for u in (sample.model_usage or {}).values())
                row["tokens_per_decision"] = tokens / len(decisions) if decisions else None
            groups[group].append(row)
    return dict(groups)


def fmt(value: float | None, percent: bool = False) -> str:
    if value is None or not finite(value):
        return "–"
    return f"{value:.1%}" if percent else f"{value:.2f}"


def table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| group | model | tools | completed / attempts | errors | pairs | win | unassisted win | total cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group, rows in sorted(groups.items()):
        config = rows[0]["config"]
        done = [r for r in rows if r["complete"]]
        estimate = pair_mean(done)
        errors = sum(bool(r["error"]) for r in rows)
        costs = [r["cost"] for r in rows]
        cost = "–" if any(c is None for c in costs) else f"${sum(costs):.3f}"
        cells = [
            group,
            config["model"],
            config["tool_access"],
            f"{len(done)} / {len(rows)}",
            str(errors),
            str(estimate["pairs"]),
            fmt(estimate["estimate"], True),
            fmt(pair_mean(done, "unassisted_win")["estimate"], True),
            cost,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Attempts are retained sample records, including errors; aborted runs may have unstarted samples.",
        "Wins are conditional on completion. Unassisted wins exclude harness defaults and substitutions.",
        "Win means weight observed unordered team pairs equally. No confidence interval is inferred",
        "from the six shared teams. Use --compare for matched outcomes and missing-outcome bounds.",
        "Different groups are different configurations or engines; unequal coverage is not a comparison.",
        "Costs include retained failed samples; discarded retries may incur additional unrecorded cost.",
        "Use --json for complete configurations, per-sample diagnostics, and source paths.",
    ]
    for group, rows in sorted(groups.items()):
        cells = [(r["sample_id"], r["epoch"]) for r in rows]
        if len(set(cells)) != len(cells):
            lines.append(
                f"WARNING {group}: repeated sample/epoch cells across logs; descriptive attempts only."
            )
        if any(r["retried"] for r in rows):
            lines.append(f"WARNING {group}: sample retries occurred; comparison is disabled.")
    return "\n".join(lines)


def comparison(groups, selection):
    left, right = selection.split(",")
    a, b = groups[left], groups[right]
    # A calculator ablation may change exactly the tool access and corresponding prompt/catalog.
    allowed = {"tool_access", "system_sha256", "tools_sha256"}
    differences = [
        k for k in a[0]["config"] if k not in allowed and a[0]["config"][k] != b[0]["config"][k]
    ]
    if differences:
        raise ValueError("incompatible ablation configurations: " + ", ".join(differences))
    if {a[0]["config"]["tool_access"], b[0]["config"]["tool_access"]} != {"full", "no_calculators"}:
        raise ValueError("comparison requires full and no_calculators conditions")
    return {"contrast": f"{right} minus {left}", **paired_contrast(a, b)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not finite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=IN,OUT[,CACHE_READ,CACHE_WRITE]",
        help="USD per million tokens; explicit cache prices required for cached usage",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable attempts and configurations"
    )
    parser.add_argument(
        "--compare", metavar="LEFT_GROUP,RIGHT_GROUP", help="matched calculator ablation"
    )
    args = parser.parse_args()
    try:
        groups = collect(args.log_dir, parse_prices(args.price))
        if not groups:
            raise ValueError("no recorded samples found in this log directory")
        contrast = comparison(groups, args.compare) if args.compare else None
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(
            json.dumps(
                json_safe({"groups": groups, "comparison": contrast}), indent=2, allow_nan=False
            )
        )
    else:
        print(table(groups))
        if contrast:
            print("\n" + json.dumps(contrast, indent=2))


if __name__ == "__main__":
    main()
