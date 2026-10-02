from types import SimpleNamespace as NS

import pytest
from inspect_ai.model import ModelUsage

from league_evals import report


def test_cost_uses_all_token_categories_and_never_invents_cache_discount():
    usage = {
        "m": ModelUsage(
            input_tokens=100,
            output_tokens=10,
            input_tokens_cache_read=1000,
            input_tokens_cache_write=20,
        )
    }
    assert report.sample_cost(usage, {"m": (2, 8)}) is None
    assert report.sample_cost(usage, {"m": (2, 8, 0.5, 3)}) == pytest.approx(0.00084)
    usage["m"].total_cost = 0.012
    assert report.sample_cost(usage, {}) == 0.012
    assert report.sample_cost({}, {}) is None


@pytest.mark.parametrize("entry", ["m=nan,1", "m=-1,1", "m=1,2,3", "=1,2", "m=1"])
def test_bad_prices_are_rejected(entry):
    with pytest.raises(ValueError):
        report.parse_prices([entry])


def fake_log(status="success", *, error=None, budget=24):
    config = NS(
        **{
            k: None
            for k in (
                "token_limit",
                "token_limit_type",
                "message_limit",
                "time_limit",
                "working_limit",
                "cost_limit",
                "retry_on_error",
            )
        }
    )
    outcome = {
        "winner": "focal",
        "turns": 3,
        "simulator_substitutions": {"p1": 0, "p2": 0},
        "decisions": {"p1": [], "p2": []},
    }
    sample = NS(
        id="a--b--1",
        epoch=1,
        metadata={"pair": "a--b", "seed": 1, "focal_team": "a", "opponent_team": "b"},
        store={"outcome": outcome},
        messages=[],
        error=error,
        invalidation=None,
        limit=None,
        error_retries=None,
        model_usage={},
    )
    spec = NS(
        model="m",
        task="vgc_battle",
        task_version=3,
        task_args={"max_generations": budget},
        model_generate_config=NS(model_dump=lambda **kw: {}),
        model_args={},
        model_base_url=None,
        config=config,
    )
    return NS(status=status, eval=spec, samples=[sample])


def test_report_keeps_errors_and_completed_samples_in_failed_runs_and_splits_configs(
    monkeypatch, tmp_path
):
    logs = [
        fake_log(),
        fake_log("error", error="provider failed"),
        fake_log("error"),
        fake_log(budget=16),
    ]
    monkeypatch.setattr(report, "list_eval_logs", lambda path: [NS(name=str(i)) for i in range(4)])
    monkeypatch.setattr(report, "read_eval_log", lambda path: logs[int(path)])
    groups = report.collect(tmp_path, {})
    assert sorted(len(rows) for rows in groups.values()) == [1, 3]
    combined = next(rows for rows in groups.values() if len(rows) == 3)
    assert sum(r["complete"] for r in combined) == 2
    assert combined[1]["error"] == "provider failed"
    assert "repeated sample/epoch" in report.table(groups)


def test_ablation_refuses_budget_confound():
    a = {"config": {"tool_access": "full", "max_generations": 24}}
    b = {"config": {"tool_access": "no_calculators", "max_generations": 16}}
    with pytest.raises(ValueError, match="max_generations"):
        report.comparison({"a": [a], "b": [b]}, "a,b")
