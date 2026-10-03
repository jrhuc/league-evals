import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai import eval_async
from inspect_ai._cli.util import parse_cli_args
from inspect_ai.model import ModelOutput
from mock_model import offline_model

from league_evals import league
from league_evals.battle import harness_tool, vgc_battle
from league_evals.bridge import BridgeRejected
from league_evals.scorers import conduct_summary, league_summary

SCHEMAS = json.loads((Path(__file__).parent / "league_schemas.json").read_text())


def test_every_stage_schema_converts_to_inspect_tools():
    async def execute(**kwargs):
        return ""

    for stage in SCHEMAS.values():
        for definition in [*stage["tools"], stage["submission"]]:
            assert harness_tool(definition, execute).name == definition["name"]


def outcome(focal_entrant=1):
    return {
        "run_dir": "/tmp/run",
        "entrants": ["bot", "external:model", "bot"],
        "team_names": ["Bot Coach 1", "Probe", "Bot Coach 3"],
        "standings": [
            {"entrant": 1, "w": 2, "l": 0, "gw": 4, "gl": 1},
            {"entrant": 0, "w": 1, "l": 1, "gw": 2, "gl": 2},
            {"entrant": 2, "w": 0, "l": 2, "gw": 1, "gl": 4},
        ],
        "series": [
            {
                "index": 0,
                "stage": "roundrobin",
                "round": 1,
                "entrants": [1, 0],
                "score": {"p1": 2, "p2": 1},
                "winner": 1,
            },
            {
                "index": 1,
                "stage": "roundrobin",
                "round": 2,
                "entrants": [2, 1],
                "score": {"p1": 0, "p2": 2},
                "winner": 1,
            },
            {
                "index": 2,
                "stage": "playoff",
                "round": 1,
                "entrants": [0, 1],
                "score": {"p1": 2, "p2": 0},
                "winner": 0,
            },
        ],
        "placement": [0, focal_entrant, 2],
        "error": None,
    }


def test_league_summary_reads_placement_series_and_games():
    summary = league_summary(outcome(), 1)
    assert summary["placement"] == 2
    assert summary["placement_score"] == 0.5
    assert summary["champion"] == 0.0
    assert summary["playoffs"] == 1.0
    assert summary["regular_season_rank"] == 1
    assert summary["series_win_rate"] == pytest.approx(2 / 3)
    assert summary["game_win_rate"] == pytest.approx(4 / 7)


def exchange(ident, stage, session, task):
    return {
        "kind": "exchange",
        "entrant": 1,
        "exchange": {
            "id": ident,
            "model": league.MODEL_SEAT,
            "session": session,
            "task": task,
            "system": "SYSTEM " + stage,
            "prompt": f"PROMPT {task}",
            "tools": SCHEMAS[stage]["tools"],
            "submission": SCHEMAS[stage]["submission"],
        },
    }


class FakeLeague:
    def __init__(self, script, rejections=0):
        self.script = list(script)
        self.rejections = rejections
        self.events = asyncio.Queue()
        self.requests = []
        self.closed = False
        self.hello = {"run_dir": "/tmp/run", "harness_commit": "h", "showdown_commit": "s"}
        self.advance()

    def advance(self):
        while self.script:
            event = self.script.pop(0)
            self.events.put_nowait(event)
            if event["kind"] == "exchange":
                break

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "tool":
            return "- garchomp | Ground/Dragon | 18"
        if method == "submit" and self.rejections:
            self.rejections -= 1
            raise BridgeRejected('"pick" must name a board id')
        if method in ("submit", "abandon"):
            self.advance()
            return {"accepted": True}
        raise AssertionError(method)

    async def next_event(self):
        return await self.events.get()

    async def close(self):
        self.closed = True


def install(monkeypatch, fake):
    async def start(*args, **kwargs):
        fake.params = args[0]
        return fake

    monkeypatch.setattr(league.LeagueBridge, "league", start)
    monkeypatch.setattr(league, "run_root", lambda: Path("/tmp/league-test-runs"))


def respond(messages, tools, *args):
    prompt = next(m.text for m in reversed(messages) if m.role == "user")
    names = {tool.name for tool in tools}
    if "submit_pick" in names:
        if not any(m.role == "tool" for m in messages):
            return ModelOutput.for_tool_call("mock", "search_board", {})
        return ModelOutput.for_tool_call("mock", "submit_pick", {"pick": "garchomp"})
    if "submit_action" in names:
        return ModelOutput.from_content("mock", f"Thinking about {prompt}")
    return ModelOutput.for_tool_call("mock", "submit_name", {"team_name": "Probe"})


async def test_a_season_runs_every_task_through_its_own_session(monkeypatch, tmp_path):
    fake = FakeLeague(
        [
            exchange(1, "submit_pick", "draft-1", "pick-2"),
            exchange(2, "submit_name", "name-1", "name-1"),
            {
                "kind": "series",
                "series": outcome()["series"][0],
            },
            exchange(3, "submit_action", "battle-g1-p1", "decision-1"),
            {"kind": "end", "outcome": outcome()},
        ],
        rejections=1,
    )
    install(monkeypatch, fake)
    logs = await eval_async(
        league.vgc_league(max_generations=3),
        model=offline_model(respond),
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert fake.closed
    assert fake.params["seats"] == [league.MODEL_SEAT, "bot", "bot", "bot"]
    methods = [method for method, _ in fake.requests]
    assert methods == ["tool", "submit", "submit", "submit", "abandon"]
    steps = sample.store["steps"]
    assert [step["tool"] for step in steps] == ["submit_pick", "submit_name", "submit_action"]
    assert steps[0]["rejected"] == 1 and steps[0]["tool_calls"] == 1
    assert steps[2]["defaulted"] and steps[2]["generations"] == 3
    assert sample.store["series"] == [outcome()["series"][0]]
    standing = sample.scores["league_standing"].value
    assert standing["placement"] == 2 and standing["completed"] == 1.0
    conduct = sample.scores["league_conduct"].value
    assert conduct["defaulted_decisions"] == 1 and conduct["battle_decisions"] == 1


async def test_a_stage_task_without_a_submission_ends_the_season(monkeypatch, tmp_path):
    fake = FakeLeague(
        [exchange(1, "submit_team", "build-1", "build-1"), {"kind": "end", "outcome": outcome()}]
    )
    install(monkeypatch, fake)
    logs = await eval_async(
        league.vgc_league(max_generations=2),
        model=offline_model(lambda *args: ModelOutput.from_content("mock", "no")),
        log_dir=str(tmp_path),
        fail_on_error=False,
    )
    sample = logs[0].samples[0]
    assert sample.error and "no accepted submission" in sample.error.message
    assert fake.closed
    assert sample.store["completion"] == "incomplete"
    assert sample.store["steps"][0]["generations"] == 2


def test_conduct_counts_steps():
    steps = [
        {
            "tool": "submit_action",
            "defaulted": True,
            "rejected": 0,
            "tool_errors": 1,
            "post_submission_calls": 0,
            "tool_calls": 2,
            "generations": 4,
        },
        {
            "tool": "submit_pick",
            "defaulted": False,
            "rejected": 2,
            "tool_errors": 0,
            "post_submission_calls": 1,
            "tool_calls": 0,
            "generations": 2,
        },
    ]
    summary = conduct_summary(steps, 100)
    assert summary["tasks"] == 2 and summary["battle_decisions"] == 1
    assert summary["defaulted_decisions"] == 1 and summary["rejected_submissions"] == 2
    assert summary["tool_calls_per_task"] == 1.0 and summary["generations_per_task"] == 3.0


def test_seed_lists_from_the_command_line_become_one_sample_per_seed():
    many = parse_cli_args(("seeds=1,2,3", "control=bot"))
    assert len(league.vgc_league(**many).dataset) == 3
    assert len(league.vgc_league(**parse_cli_args(("seeds=4",))).dataset) == 1
    one_seed = len(vgc_battle(**parse_cli_args(("seeds=1",))).dataset)
    assert len(vgc_battle(**parse_cli_args(("seeds=1,2",))).dataset) == 2 * one_seed
