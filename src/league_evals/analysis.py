"""Transparent finite-pool comparisons; missing outcomes are unknown, never losses."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def pair_mean(rows: list[dict[str, Any]], key: str = "win") -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("pair") and finite(row.get(key)):
            groups[row["pair"]].append(row[key])
    by_pair = {pair: mean(groups[pair]) for pair in sorted(groups)}
    return {
        "estimate": mean(by_pair.values()) if by_pair else None,
        "pairs": len(by_pair),
        "by_pair": by_pair,
    }


def _index(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    indexed = {}
    fields = ("focal_team", "opponent_team", "focal_seat", "seed", "epoch")
    for row in rows:
        if any(row.get(field) is None for field in fields) or not row.get("pair"):
            raise ValueError("comparison needs team, seat, seed, epoch, and pair metadata")
        if row.get("retried"):
            raise ValueError("comparison needs attempts without sample retries; rerun a full block")
        key = tuple(row[field] for field in fields)
        # Check before filtering failures: failed + successful retries are still duplicates.
        if key in indexed:
            raise ValueError("duplicate experimental cell; retain one declared attempt per cell")
        if row.get("complete") and row.get("win") not in (0, 1):
            raise ValueError("completed cells need a binary win outcome")
        indexed[key] = row
    return indexed


def paired_contrast(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    """Right minus left. Bounds cover cells observed in either arm, including failures."""
    a, b = _index(left), _index(right)
    differences, bounds = [], []
    right_only_wins = left_only_wins = same_outcome = 0
    for key in sorted(a.keys() | b.keys()):
        ar, br = a.get(key, {}), b.get(key, {})
        if ar and br and ar["pair"] != br["pair"]:
            raise ValueError("matching cells have inconsistent pair metadata")
        pair = (ar or br)["pair"]
        aw = ar["win"] if ar.get("complete") else None
        bw = br["win"] if br.get("complete") else None
        if aw is not None and bw is not None:
            differences.append({"pair": pair, "win": bw - aw})
            right_only_wins += bw > aw
            left_only_wins += aw > bw
            same_outcome += aw == bw
        bounds.append(
            {
                "pair": pair,
                "lower": (bw if bw is not None else 0) - (aw if aw is not None else 1),
                "upper": (bw if bw is not None else 1) - (aw if aw is not None else 0),
            }
        )
    return {
        **pair_mean(differences),
        "matched_cells": len(differences),
        "observed_cells": len(bounds),
        "right_only_wins": right_only_wins,
        "left_only_wins": left_only_wins,
        "same_outcome": same_outcome,
        "left_missing": len(b.keys() - a.keys()),
        "right_missing": len(a.keys() - b.keys()),
        "left_incomplete": sum(not r.get("complete") for r in left),
        "right_incomplete": sum(not r.get("complete") for r in right),
        "missing_outcome_bounds": [pair_mean(bounds, k)["estimate"] for k in ("lower", "upper")],
        "bounds_note": "Worst/best outcomes for missing or unfinished cells observed in either arm. "
        "Not a confidence interval; cells absent from both arms are unknown.",
    }
