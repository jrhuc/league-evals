from collections import deque
from dataclasses import dataclass, field

import pytest
from fake_bridge import SEATS, SYSTEM, decision_row, exchange_event, external
from inspect_ai import eval_async
from inspect_ai.model import ModelOutput
from inspect_ai.tool import ToolCall
from mock_model import offline_model

from league_evals import battle
from league_evals.bridge import BridgeError
from league_evals.scorers import EMPTY_AUDIT

REJECTION = "|error|[Invalid choice] Can't move: the move is disabled"


@dataclass
class FakeBridge:
    fail: bool = False
    rejections: int = 0
    error: str | None = None
    count: int = 0
    sent: int = 0
    closed: bool = False
    focal: str = "p1"
    requests: list = field(default_factory=list)
    events: deque = field(default_factory=deque)
    rows: list = field(default_factory=list)
    hello: dict = field(
        default_factory=lambda: {**battle.load_pool("test")["provenance"], "seats": SEATS}
    )

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "start":
            self.focal = external(params)
            self.events.append(self.exchange())
            return {"started": True}
        if method in ("submit", "abandon"):
            if self.fail:
                raise BridgeError("simulator transport failed")
            source = "model" if method == "submit" else "model-default"
            if self.rejections:
                self.rejections -= 1
                self.resolve(
                    decision_row(self.focal, "move 1", outcome="rejected", error=REJECTION)
                )
                self.events.append(self.exchange())
                return {"accepted": True}
            self.count += 1
            self.resolve(decision_row(self.focal, "move 1", source=source))
            self.events.append(self.exchange() if self.count < 2 else self.end())
            return {"accepted": True}
        if method == "tool":
            return "Reference result"
        if method == "audit":
            return EMPTY_AUDIT
        raise AssertionError(method)

    async def next_event(self):
        return self.events.popleft()

    def resolve(self, event):
        self.rows.append(event["row"])
        self.events.append(event)

    def exchange(self):
        self.sent += 1
        return exchange_event(
            self.focal,
            self.sent,
            turn=self.count,
            phase="team_preview" if self.count == 0 else "turn",
            names=("estimate_damage", "compare_action_order", "lookup_species"),
        )

    def end(self):
        return {
            "kind": "end",
            "outcome": {
                "winner": "focal",
                "turns": 1,
                "log": ["|win|focal"],
                "simulator_substitutions": {"p1": 0, "p2": 0},
                "decisions": {"p1": [], "p2": [], self.focal: self.rows},
                "error": self.error,
            },
        }

    async def close(self):
        self.closed = True


def install_bridge(monkeypatch, bridge):
    async def open_bridge(*args, **kwargs):
        return bridge

    monkeypatch.setattr(battle.LeagueBridge, "open", open_bridge)


def submit():
    return ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0], "rationale": "why"})


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
        output = submit()
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
    assert not any(m == "tool" for m, _ in bridge.requests)
    start = next(p for m, p in bridge.requests if m == "start")
    assert start["p2"] == {
        "name": "focal",
        "team": sample.metadata["focal_packed"],
        "seat": "external",
    }
    assert start["p1"]["seat"] == "search" and start["p1"]["name"] == "opponent"
    assert start["policy_seed"] == start["seed"] == sample.metadata["seed"]
    submissions = [p for m, p in bridge.requests if m == "submit"]
    assert [(p["pid"], p["exchange"]) for p in submissions] == [("p2", 1), ("p2", 2)]
    assert all(p["input"] == {"choices": [0], "rationale": "why"} for p in submissions)
    assert [p["usage"]["input_tokens"] for p in submissions] == [20, 10]
    decisions = sample.store["decisions"]
    assert decisions[0]["rejected"] == 2  # Schema rejection plus duplicate submission.
    assert [d["post_submission_calls"] for d in decisions] == [2, 2]
    assert [(d["number"], d["turn"], d["phase"]) for d in decisions] == [
        (1, 0, "team_preview"),
        (2, 1, "turn"),
    ]
    assert all(d["rationale"] == "why" for d in decisions)
    assert sample.store["completion"] == "complete"
    assert sample.store["system_prompt"].startswith(SYSTEM + "\nEach decision allows up to 3")
    users = [m.text for m in sample.messages if m.role == "user"]
    assert users[:2] == [battle.SAMPLE_INPUT, "Choose index 0."]


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
    assert [p["exchange"] for m, p in bridge.requests if m == "abandon"] == [1, 2]
    assert sample.scores["outcome"].value["win"] == 1
    assert sample.scores["outcome"].value["unassisted_win"] == 0
    assert all(d["defaulted"] for d in sample.store["decisions"])
    assert sample.store["system_prompt"].endswith(battle.NO_CALCULATORS)
    assert [t["name"] for t in sample.store["tool_catalog"]] == ["lookup_species"]


async def test_showdown_rejection_reopens_the_decision(monkeypatch, tmp_path):
    bridge = FakeBridge(rejections=1)
    install_bridge(monkeypatch, bridge)
    logs = await eval_async(
        battle.vgc_battle(),
        model=offline_model(lambda *args: submit()),
        limit=1,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    decisions = sample.store["decisions"]
    assert [(d["number"], d["turn"], d["error"]) for d in decisions] == [
        (1, 0, None),
        (2, 0, REJECTION),
        (3, 1, None),
    ]
    assert sample.scores["outcome"].value["showdown_rejections"] == 1
    assert sample.scores["outcome"].value["unassisted_win"] == 1


async def test_engine_failure_preserves_partial_trace_and_is_not_a_model_error(
    monkeypatch, tmp_path
):
    bridge = FakeBridge(fail=True)
    install_bridge(monkeypatch, bridge)
    logs = await eval_async(
        battle.vgc_battle(),
        model=offline_model([submit()]),
        limit=1,
        fail_on_error=False,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is not None
    assert bridge.closed
    assert sample.store["failure"]["type"] == "BridgeError"
    assert len(sample.store["decisions"]) == 1
    assert sample.store["decisions"][0]["rejected"] == 0
    assert sample.store["completion"] == "incomplete"


@pytest.mark.parametrize("failure", ["harness", "opponent"])
async def test_harness_failure_and_unknown_opponent_are_evaluation_errors(
    monkeypatch, tmp_path, failure
):
    bridge = FakeBridge(error="p2 chose an action its request does not offer")
    install_bridge(monkeypatch, bridge)
    logs = await eval_async(
        battle.vgc_battle(opponent="greedy" if failure == "harness" else "external"),
        model=offline_model(lambda *args: submit()),
        limit=1,
        fail_on_error=False,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is not None and bridge.closed
    assert sample.store["failure"]["type"] == "BridgeError"
    assert sample.store["completion"] == "incomplete"
    if failure == "harness":
        assert "does not offer" in sample.store["failure"]["message"]
        assert sample.store["outcome"]["winner"] == "focal"
    else:
        assert "greedy, search" in sample.store["failure"]["message"]
        assert not bridge.requests


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
        {"max_generations": 0},
        {"max_decisions": 0},
        {"tool_access": "other"},
        {"seeds": ""},
        {"seeds": "1,1"},
        {"focal_seat": "other"},
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
        assert messages[0].text.endswith(battle.NO_CALCULATORS) != calculators
        return submit()

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
