"""Outcomes, calculator diagnostics, observable tool use, and resource consumption."""

from __future__ import annotations

import re
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

CALCULATORS = frozenset({"estimate_damage", "compare_action_order"})
FOCAL_NAME = "focal"
UNDEFINED = float("nan")
"""Inspect skips NaN-valued keys when it aggregates a dict score, so a rate with no
observations stays out of the mean instead of counting as zero."""


def outcome_summary(
    outcome: dict[str, Any], focal_seat: str = "p1", defaulted: bool = False
) -> dict[str, Any]:
    focal = [row for row in outcome["decisions"][focal_seat] if row["kind"] == "decision"]
    substitutions = outcome["simulator_substitutions"][focal_seat]
    assisted = (
        defaulted
        or substitutions > 0
        or any(
            row["submission_source"] in {"model-default", "simulator-default", "timer-default"}
            for row in focal
        )
    )
    win = float(outcome["winner"] == FOCAL_NAME)
    return {
        "completed": 1.0,
        "win": win,
        "unassisted_win": float(bool(win) and not assisted),
        "assisted": float(assisted),
        "draw": float(outcome["winner"] is None),
        "turns": outcome["turns"],
        "showdown_rejections": sum(1 for row in focal if row["outcome"] == "rejected"),
        "simulator_substitutions": substitutions,
    }


def mechanics_summary(audit: dict[str, Any]) -> dict[str, Any]:
    def rate(matched: int, total: int) -> float:
        return matched / total if total else UNDEFINED

    damage_live = audit["damagePredictions"] - audit.get("damageHypothetical", 0)
    order_live = audit["orderPredictions"] - audit.get("orderHypothetical", 0)
    return {
        "damage_predictions": audit["damagePredictions"],
        "damage_paired": audit["damageMatched"],
        "damage_pair_rate": rate(audit["damageMatched"], damage_live),
        "order_predictions": audit["orderPredictions"],
        "order_paired": audit["orderMatched"],
        "order_pair_rate": rate(audit["orderMatched"], order_live),
        "hypothetical_predictions": (
            audit.get("damageHypothetical", 0) + audit.get("orderHypothetical", 0)
        ),
        "findings": len(audit["findings"]),
        # Findings are deduplicated while matched calls are not; a match can have
        # multiple findings. This is a diagnostic density, never an error probability.
        "findings_per_match": rate(
            len(audit["findings"]), audit["damageMatched"] + audit["orderMatched"]
        ),
    }


def discipline_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(decisions)
    calls = sum(len(d["calls"]) for d in decisions)
    calculated = sum(1 for d in decisions if any(c in CALCULATORS for c in d["calls"]))
    preview_calcs = sum(
        sum(1 for c in d["calls"] if c in CALCULATORS)
        for d in decisions
        if d["phase"] == "team_preview"
    )
    return {
        "decisions": total,
        "preview_calcs": preview_calcs,
        "calc_decision_rate": calculated / total if total else UNDEFINED,
        "tool_calls_per_decision": calls / total if total else UNDEFINED,
        "generations_per_decision": (
            sum(d["generations"] for d in decisions) / total if total else UNDEFINED
        ),
        "rejected_submissions": sum(d["rejected"] for d in decisions),
        "defaulted_decisions": sum(1 for d in decisions if d["defaulted"]),
        "post_submission_calls": sum(d.get("post_submission_calls", 0) for d in decisions),
        "tool_errors": sum(d.get("tool_errors", 0) for d in decisions),
    }


def efficiency_summary(total_tokens: int, decisions: int) -> dict[str, Any]:
    return {
        "total_tokens": total_tokens,
        "tokens_per_decision": total_tokens / decisions if decisions else UNDEFINED,
    }


def _explain(summary: dict[str, Any]) -> str:
    return ", ".join(
        f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()
    )


@scorer(
    metrics={
        "completed": [mean()],
        "win": [mean()],
        "unassisted_win": [mean()],
        "assisted": [mean()],
        "draw": [mean()],
        "turns": [mean()],
        "showdown_rejections": [mean()],
        "simulator_substitutions": [mean()],
    }
)
def outcome() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        result = state.store.get("outcome")
        if not result or state.store.get("completion", "complete") != "complete":
            return Score(
                value={
                    "completed": 0.0,
                    "win": UNDEFINED,
                    "unassisted_win": UNDEFINED,
                    "assisted": UNDEFINED,
                    "draw": UNDEFINED,
                    "turns": UNDEFINED,
                    "showdown_rejections": UNDEFINED,
                    "simulator_substitutions": UNDEFINED,
                },
                reason="no_response",
                explanation="no completed game",
            )
        summary = outcome_summary(
            result,
            state.metadata.get("focal_seat", "p1"),
            any(d["defaulted"] for d in state.store.get("decisions", [])),
        )
        return Score(value=summary, answer=result["winner"], explanation=_explain(summary))

    return score


EMPTY_AUDIT: dict[str, Any] = {
    "damagePredictions": 0,
    "damageMatched": 0,
    "damageHypothetical": 0,
    "orderPredictions": 0,
    "orderMatched": 0,
    "orderHypothetical": 0,
    "findings": [],
}


@scorer(
    metrics={
        "damage_predictions": [mean()],
        "damage_paired": [mean()],
        "damage_pair_rate": [mean()],
        "order_predictions": [mean()],
        "order_paired": [mean()],
        "order_pair_rate": [mean()],
        "hypothetical_predictions": [mean()],
        "findings": [mean()],
        "findings_per_match": [mean()],
    }
)
def mechanics() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        audit = state.store.get("audit")
        if not audit:
            return Score(value=mechanics_summary(EMPTY_AUDIT), explanation="no audit")
        summary = mechanics_summary(audit)
        findings = "; ".join(f["detail"] for f in audit["findings"][:5])
        return Score(
            value=summary, explanation=_explain(summary) + (f" | {findings}" if findings else "")
        )

    return score


@scorer(
    metrics={
        "decisions": [mean()],
        "preview_calcs": [mean()],
        "calc_decision_rate": [mean()],
        "tool_calls_per_decision": [mean()],
        "generations_per_decision": [mean()],
        "rejected_submissions": [mean()],
        "defaulted_decisions": [mean()],
        "post_submission_calls": [mean()],
        "tool_errors": [mean()],
    }
)
def discipline() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        summary = discipline_summary(state.store.get("decisions", []))
        return Score(value=summary, explanation=_explain(summary))

    return score


@scorer(metrics={"total_tokens": [mean()], "tokens_per_decision": [mean()]})
def efficiency() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        decisions = len(state.store.get("decisions", []))
        summary = efficiency_summary(state.token_usage, decisions)
        return Score(value=summary, explanation=_explain(summary))

    return score


@scorer(
    metrics={
        key: [mean()]
        for key in (
            "value",
            "regret",
            "good",
            "beats_greedy",
            "matches_recorded",
            "scored",
            "on_demand",
            "defaulted",
        )
    }
)
def position_regret() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        import asyncio

        from .positions import regret, value_command

        position = state.metadata["position"]
        command = state.store.get("position_choice")
        entry = position["values"].get(command)
        on_demand = bool(command and command != "forfeit" and entry is None)
        if on_demand:
            entry = await asyncio.to_thread(
                value_command, state.metadata["dataset"], position, command
            )
        result = regret(position, command, entry) if command else None
        summary = {
            **(
                result
                or {
                    key: UNDEFINED
                    for key in ("value", "regret", "good", "beats_greedy", "matches_recorded")
                }
            ),
            "scored": float(result is not None),
            "on_demand": float(on_demand),
            "defaulted": float(any(d["defaulted"] for d in state.store.get("decisions", []))),
        }
        return Score(
            value=summary, answer=command, explanation=_explain(summary), metadata={"entry": entry}
        )

    return score


@scorer(metrics={key: [mean()] for key in ("verified", "any_calculator", "followed", "defaulted")})
def note_trust() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        claim = state.metadata["note"]["claim"]
        pattern = re.compile(
            rf"^{re.escape(claim['attacker'])}(-Mega\S*)? {re.escape(claim['move'])} "
            rf"\(.*\) into {re.escape(claim['defender'])}(-Mega\S*)?:",
            re.MULTILINE,
        )
        decisions = state.store.get("decisions", [])
        calls = [
            call
            for decision in decisions
            for call in decision.get("trace", [])
            if "result" in call and "error" not in call
        ]
        command = state.store.get("position_choice")
        parts = command.split(", ") if command else []
        slot = claim["slot"]
        summary = {
            "verified": float(
                any(
                    call["name"] == "estimate_damage" and pattern.search(call["result"])
                    for call in calls
                )
            ),
            "any_calculator": float(any(call["name"] in CALCULATORS for call in calls)),
            "followed": float(
                slot < len(parts) and parts[slot].removesuffix(" mega") == claim["follow_part"]
            )
            if command
            else UNDEFINED,
            "defaulted": float(any(d["defaulted"] for d in decisions)),
        }
        return Score(value=summary, answer=command, explanation=_explain(summary))

    return score
