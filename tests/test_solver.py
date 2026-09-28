from dataclasses import dataclass, field

import pytest
from inspect_ai import eval_async
from inspect_ai.model import ModelOutput
from inspect_ai.tool import ToolCall
from mock_model import offline_model

from league_evals import battle
from league_evals.bridge import BridgeError
from league_evals.scorers import EMPTY_AUDIT


@dataclass
class FakeBridge:
    fail: bool = False
    count: int = 0
    closed: bool = False
    requests: list = field(default_factory=list)
    hello: dict = field(default_factory=lambda: battle.load_pool("test")["provenance"])

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "tools":
            return [
                {
                    "name": n,
                    "description": "Reference lookup.",
                    "parameters": {"type": "object", "properties": {}},
                }
                for n in ("estimate_damage", "compare_action_order", "lookup_species")
            ]
        if method == "start":
            return self.event()
        if method == "submit":
            if self.fail:
                raise BridgeError("simulator transport failed")
            self.count += 1
            if self.count < 2:
                return {"choice": "move 1", "next": self.event()}
            return {
                "choice": "move 1",
                "next": {
                    "kind": "end",
                    "outcome": {
                        "winner": "focal",
                        "turns": 1,
                        "log": ["|win|focal"],
                        "submissions": {"p1": [], "p2": []},
                        "simulator_substitutions": {"p1": 0, "p2": 0},
                    },
                },
            }
        if method == "call":
            return "Reference result"
        if method == "audit":
            return EMPTY_AUDIT
        raise AssertionError(method)

    def event(self):
        return {
            "kind": "decision",
            "decision": self.count + 1,
            "turn": self.count,
            "phase": "team_preview" if self.count == 0 else "turn",
            "error": None,
            "prompt": "Choose index 0.",
        }

    async def close(self):
        self.closed = True


def install_bridge(monkeypatch, bridge):
    async def open_bridge(*args, **kwargs):
        return bridge

    monkeypatch.setattr(battle.LeagueBridge, "open", open_bridge)


async def test_one_reply_cannot_advance_multiple_decisions_or_read_unseen_state(
    monkeypatch, tmp_path
):
    bridge = FakeBridge()
    install_bridge(monkeypatch, bridge)
    calls = 0

    def respond(*args):
        nonlocal calls
        calls += 1
        # Inspect rejects this before the tool body; it must still be counted.
        if calls == 1:
            return ModelOutput.for_tool_call("mock", "submit_action", {"choices": "invalid"})
        output = ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0]})
        output.message.tool_calls.extend(
            [
                ToolCall(id=f"late-{calls}", function="lookup_species", arguments={}),
                ToolCall(
                    id=f"duplicate-{calls}", function="submit_action", arguments={"choices": [0]}
                ),
            ]
        )
        return output

    logs = await eval_async(
        battle.vgc_battle(focal_seat="p2", max_generations=3),
        model=offline_model(respond),
        limit=1,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert bridge.closed and bridge.count == 2
    assert not any(m == "call" for m, _ in bridge.requests)
    submissions = [p for m, p in bridge.requests if m == "submit"]
    assert all(p["pid"] == "p2" for p in submissions)
    decisions = sample.store["decisions"]
    assert decisions[0]["rejected"] == 2  # Schema rejection plus duplicate submission.
    assert [d["post_submission_calls"] for d in decisions] == [2, 2]
    assert sample.store["completion"] == "complete"


async def test_ablation_removes_tools_and_records_assistance(monkeypatch, tmp_path):
    bridge = FakeBridge()
    install_bridge(monkeypatch, bridge)

    def respond(messages, tools, *args):
        assert {t.name for t in tools} == {"lookup_species", "submit_action"}
        return ModelOutput.from_content("mock", "I did not submit.")

    logs = await eval_async(
        battle.vgc_battle(tool_access="no_calculators", max_generations=1),
        model=offline_model(respond),
        limit=1,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert sample.scores["outcome"].value["win"] == 1
    assert sample.scores["outcome"].value["unassisted_win"] == 0
    assert all(d["defaulted"] for d in sample.store["decisions"])
    assert "unavailable" in sample.store["system_prompt"]


async def test_engine_failure_preserves_partial_trace_and_is_not_a_model_error(
    monkeypatch, tmp_path
):
    bridge = FakeBridge(fail=True)
    install_bridge(monkeypatch, bridge)
    model = offline_model(
        [ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0]})],
    )
    logs = await eval_async(
        battle.vgc_battle(), model=model, limit=1, fail_on_error=False, log_dir=str(tmp_path)
    )
    sample = logs[0].samples[0]
    assert sample.error is not None
    assert bridge.closed
    assert sample.store["failure"]["type"] == "BridgeError"
    assert len(sample.store["decisions"]) == 1
    assert sample.store["decisions"][0]["rejected"] == 0
    assert sample.store["completion"] == "incomplete"


async def test_decision_limit_is_incomplete_not_a_loss(monkeypatch, tmp_path):
    import math

    bridge = FakeBridge()
    install_bridge(monkeypatch, bridge)
    logs = await eval_async(
        battle.vgc_battle(max_generations=1, max_decisions=1),
        model=offline_model(),
        limit=1,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert sample.store["completion"] == "decision_limit"
    assert sample.scores["outcome"].value["completed"] == 0
    assert math.isnan(sample.scores["outcome"].value["win"])
    assert bridge.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sheets": "closed"},
        {"max_generations": 0},
        {"max_decisions": 0},
        {"tool_access": "other"},
        {"seeds": ""},
        {"seeds": "1,1"},
        {"focal_seat": "other"},
        {"league_prompt": True, "tool_access": "no_calculators"},
    ],
)
def test_invalid_or_unsupported_experiments_fail_before_generation(kwargs):
    with pytest.raises(ValueError):
        battle.vgc_battle(**kwargs)


async def test_one_eval_runs_both_arms_and_report_matches_them(monkeypatch, tmp_path):
    from inspect_ai.dataset import MemoryDataset

    from league_evals.report import collect, comparison

    async def open_bridge(*args, **kwargs):
        return FakeBridge()

    monkeypatch.setattr(battle.LeagueBridge, "open", open_bridge)
    seen = set()

    def respond(messages, tools, *args):
        calculators = "estimate_damage" in {t.name for t in tools}
        seen.add(calculators)
        assert ("unavailable" in messages[0].text) != calculators
        return ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0]})

    task = battle.vgc_battle(tool_access="both")
    # First two dataset entries are both arms of the same experimental cell.
    task.dataset = MemoryDataset(list(task.dataset)[:2], name="test")
    logs = await eval_async(
        task,
        model=offline_model(respond),
        sample_shuffle=1729,
        max_samples=1,
        log_dir=str(tmp_path),
    )
    assert all(s.error is None for s in logs[0].samples)
    assert seen == {True, False}
    groups = collect(tmp_path, {})
    assert len(groups) == 2
    ids = {rows[0]["config"]["tool_access"]: group for group, rows in groups.items()}
    result = comparison(groups, f"{ids['no_calculators']},{ids['full']}")
    assert result["matched_cells"] == 1
    assert result["estimate"] == 0
    assert result["missing_outcome_bounds"] == [0, 0]
