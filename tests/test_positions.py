import copy
import json
import math
import sqlite3
import sys
from collections import deque
from dataclasses import dataclass, field

import pytest
from fake_bridge import SEATS, decision_row, exchange_event, external
from inspect_ai import eval_async
from inspect_ai.model import ModelOutput
from mock_model import offline_model
from test_bridge import needs_harness

from league_evals import position, positions
from league_evals.battle import load_pool
from league_evals.bridge import FORMAT, BridgeError, LeagueBridge, request_sync


def action(command, value, explored=None):
    return {
        "command": command,
        "value": value,
        "explored": value if explored is None else explored,
        "label": command,
    }


def game():
    return {
        "id": "g-1",
        "models": {"p1": "a", "p2": "b"},
        "log": ["log"],
        "source": {
            "format": "test",
            "seed": [1, 2, 3, 4],
            "names": {"p1": "p1-a", "p2": "p2-b"},
            "packed": {"p1": "one", "p2": "two"},
            "choices": {
                "p1": ["team 1234", "greedy", "recorded"],
                "p2": ["team 1234", "other", "other"],
            },
        },
    }


def passes():
    row = {
        "kind": "position",
        "id": "g-1",
        "focal": "p1",
        "choice_index": 2,
        "opponent_choice_index": 2,
        "turn": 2,
        "greedy": "greedy",
        "recorded": "recorded",
        "hidden_opponents": 0,
        "actions": [
            action("greedy", 0.1),
            action("recorded", 0.2),
            action("a", 0.8),
            action("b", 0.7),
        ],
    }
    choose = {
        **row,
        "actions": [
            action("greedy", 0.1),
            action("recorded", 0.2),
            action("a", 0.8),
            action("b", 0.6),
        ],
    }
    score = {
        **row,
        "actions": [
            action("greedy", 0.2),
            action("recorded", 0.3),
            action("a", 0.6),
            action("b", 0.9),
        ],
    }
    return [row], [choose], [copy.deepcopy(score)], [score]


def dataset():
    coarse, choose, select, score = passes()
    return {
        "id": "test",
        "format": "test",
        "provenance": {"harness_commit": "h", "showdown_commit": "s"},
        "selection": {
            "fine": {"samples": 24, "epsilon": 0.25, "maxTurns": 40},
            "salts": {"score": 13},
            "good_margin": 0.1,
        },
        "games": {"g-1": {k: v for k, v in game().items() if k != "id"}},
        "positions": positions.select_positions([game()], coarse, choose, select, score),
    }


def test_shortlist_keeps_greedy_and_recorded_and_drops_invalid_decisions():
    coarse, _, _, _ = passes()
    assert positions.shortlists(coarse, shortlist=1)["g-1"][0]["commands"] == [
        "a",
        "greedy",
        "recorded",
    ]
    coarse[0]["actions"] = [action("a", 1), action("b", 0)]
    assert positions.shortlists(coarse) == {}
    coarse[0]["actions"] = [action("greedy", 0.1)]
    assert positions.shortlists(coarse) == {}


def test_independent_reference_signed_regret_and_baselines():
    data = dataset()
    p = data["positions"][0]
    assert p["best"] == "a"
    assert p["good"] == ["a"]
    assert positions.regret(p, "b")["regret"] == pytest.approx(-0.3)
    assert positions.regret(p, "b")["good"] == 1
    assert positions.regret(p, "missing") is None
    assert positions.regret(p, "forfeit")["regret"] == pytest.approx(0.6)
    assert positions.regret(p, "missing", {"value": 0.7})["beats_greedy"] == 1
    rows = positions.baselines(data)
    assert rows["greedy"]["mean_regret"] == pytest.approx(0.4)
    assert rows["recorded"]["mean_regret"] == pytest.approx(0.3)
    assert rows["recorded_models"]["a"] == rows["recorded"]
    assert rows["uniform"]["mean_regret"] == pytest.approx(0.6 - (0.1 + 0.2 + 0.8 + 0.7) / 4)
    assert rows["uniform"]["n"] == 1
    assert rows["uniform"]["scored"] == 1


@pytest.mark.parametrize("change", ["missing", "single", "explored", "gap"])
def test_selection_filters(change):
    coarse, choose, select, score = passes()
    if change == "missing":
        coarse[0]["greedy"] = "absent"
    elif change == "single":
        coarse[0]["actions"] = [action("greedy", 0.1)]
    else:
        select[0]["actions"][2]["explored" if change == "explored" else "value"] = 0.25
    assert positions.select_positions([game()], coarse, choose, select, score) == []


def test_ties_and_good_margin():
    coarse, choose, select, score = passes()
    choose[0]["actions"][3]["value"] = 0.8
    selected = positions.select_positions(
        [game()], coarse, choose, select, score, good_margin=0.03
    )[0]
    assert selected["best"] == "a"
    assert selected["good"] == ["a", "b"]
    assert positions.regret(selected, "new", {"value": 0.56})["good"] == 0


def test_read_run_filters_attempts_outcomes_and_games(tmp_path):
    identity = {
        "players": {"p1": "a", "p2": "b"},
        "packed_teams": {"p1": "one", "p2": "two"},
        "format": "test",
    }
    with sqlite3.connect(tmp_path / "league.sqlite") as db:
        db.execute("CREATE TABLE series(series_id TEXT, series_index INTEGER, identity_json TEXT)")
        db.execute(
            "CREATE TABLE series_games(series_id, game_number, attempt_id, seed_json, result_json, log_path)"
        )
        db.execute("INSERT INTO series VALUES (?, ?, ?)", ("g", 0, json.dumps(identity)))
        db.execute(
            "INSERT INTO series_games VALUES (?, ?, ?, ?, ?, ?)",
            ("g", 1, "ok", "[1,2,3,4]", "{}", "game.log"),
        )
    (tmp_path / "game.log").write_text("one\ntwo\n")
    directory = tmp_path / "series/g"
    directory.mkdir(parents=True)
    base = {
        "kind": "decision",
        "game_number": 1,
        "attempt_id": "ok",
        "outcome": "accepted",
        "action": "team 1234",
    }
    rows = [
        base,
        {**base, "attempt_id": "old"},
        {**base, "outcome": "rejected"},
        {**base, "game_number": 2},
        {**base, "kind": "other"},
        {**base, "action": "move 1"},
    ]
    for pid in ("p1", "p2"):
        (directory / f"{pid}-decisions.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (result,) = positions.read_run(tmp_path)
    assert result["id"] == "g-1"
    assert result["log"] == ["one", "two", ""]
    assert result["models"] == identity["players"]
    assert result["source"]["names"] == {"p1": "p1-a", "p2": "p2-b"}
    assert result["source"]["choices"] == {pid: ["team 1234", "move 1"] for pid in ("p1", "p2")}


def test_value_games_sharding_cache_and_only(monkeypatch, tmp_path):
    calls = []

    def shard(inputs, directory):
        calls.append(inputs)
        return [{"kind": "game", "id": g["id"], "verified": True} for g in inputs]

    monkeypatch.setattr(positions, "_value_shard", shard)
    games = [{**game(), "id": str(i)} for i in range(4)]
    settings = {"samples": 4, "salt": 1}
    first = positions.value_games(games, settings, 2, cache=tmp_path)
    assert len(calls) == 2
    assert all("models" not in g for batch in calls for g in batch)
    assert positions.value_games(games, settings, 2, cache=tmp_path) == first
    assert len(calls) == 2
    only = {"1": [{"pid": "p1", "choice_index": 2, "commands": ["a"]}]}
    assert len(positions.value_games(games, settings, 2, only=only, cache=tmp_path)) == 1
    assert calls[-1][0]["only"] == only["1"]
    only["1"][0]["commands"] = ["b"]
    positions.value_games(games, settings, 2, only=only, cache=tmp_path)
    assert len(calls) == 4
    positions.value_games(games[:2], settings, 2, cache=tmp_path)
    assert len(calls) == 6


def test_value_command_uses_scoring_settings(monkeypatch):
    data = dataset()
    p = data["positions"][0]

    def value(games, settings, jobs, only, league_dir):
        assert games == [game()]
        assert settings == {"samples": 24, "epsilon": 0.25, "maxTurns": 40, "salt": 13}
        assert only == {"g-1": [{"pid": "p1", "choice_index": 2, "commands": ["new"]}]}
        assert jobs == 1
        return [
            {
                "kind": "position",
                "id": "g-1",
                "focal": "p1",
                "choice_index": 2,
                "actions": [action("new", 0.7)],
            }
        ]

    monkeypatch.setattr(positions, "value_games", value)
    assert positions.value_command(data, p, "new") == {
        "value": 0.7,
        "explored": 0.7,
        "label": "new",
    }


@dataclass
class FakeBridge:
    choice: str = "a"
    unexpected: bool = False
    fail: bool = False
    rejections: int = 0
    turn: int = 2
    count: int = 0
    sent: int = 0
    closed: bool = False
    focal: str = "p1"
    requests: list = field(default_factory=list)
    events: deque = field(default_factory=deque)
    hello: dict = field(default_factory=lambda: {**dataset()["provenance"], "seats": SEATS})

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "start":
            self.focal = external(params)
            end = {"kind": "end", "outcome": {"error": "recorded choice was rejected"}}
            self.events.append(end if self.unexpected else self.exchange())
            return {"started": True}
        if method in ("submit", "abandon"):
            if self.fail:
                raise BridgeError("transport failed")
            self.count += 1
            source = "model" if method == "submit" else "model-default"
            outcome = "rejected" if self.count <= self.rejections else "accepted"
            self.events.append(
                decision_row(self.focal, self.choice, source=source, outcome=outcome)
            )
            self.events.append(self.exchange())
            return {"accepted": True}
        raise AssertionError(method)

    async def next_event(self):
        return self.events.popleft()

    def exchange(self):
        self.sent += 1
        return exchange_event(
            self.focal,
            self.sent,
            turn=self.turn,
            names=("lookup_species", "estimate_damage", "compare_action_order"),
        )

    async def close(self):
        self.closed = True


def install(monkeypatch, bridge):
    monkeypatch.setattr(position, "load_positions", lambda name: dataset())

    async def open_bridge(*args):
        return bridge

    monkeypatch.setattr(position.LeagueBridge, "open", open_bridge)


@pytest.mark.parametrize("defaulted", [False, True])
async def test_solver_one_decision_and_report(monkeypatch, tmp_path, defaulted):
    bridge = FakeBridge()
    install(monkeypatch, bridge)
    monkeypatch.setattr(positions, "load_positions", lambda name: dataset())
    model = (
        offline_model()
        if defaulted
        else offline_model([ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0]})])
    )
    logs = await eval_async(
        position.vgc_position(positions="test", max_generations=1, tool_access="no_calculators"),
        model=model,
        log_dir=str(tmp_path),
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert bridge.count == 1 and bridge.closed
    start = next(p for m, p in bridge.requests if m == "start")
    assert start == {
        "seed": [1, 2, 3, 4],
        "p1": {"name": "focal", "team": "one", "seat": "external"},
        "p2": {"name": "opponent", "team": "two", "seat": "greedy"},
        "script": {pid: game()["source"]["choices"][pid][:2] for pid in ("p1", "p2")},
    }
    assert sample.store["position_choice"] == "a"
    assert sample.store["completion"] == "complete"
    assert len(sample.store["decisions"]) == 1
    assert sample.scores["position_regret"].value["defaulted"] == float(defaulted)
    assert sample.scores["position_regret"].value["scored"] == 1
    assert sample.scores["position_regret"].value["regret"] == 0
    assert set(sample.scores) == {"position_regret", "discipline", "efficiency"}
    assert [t["name"] for t in sample.store["tool_catalog"]] == ["lookup_species"]
    result = positions.report(tmp_path)
    assert result["groups"][0]["paired"] == {"better": 1, "same": 0, "worse": 0}
    assert result["groups"][0]["defaulted"] == int(defaulted)
    assert "coarse estimate" in positions.report_table(result)


async def test_solver_replays_a_decision_showdown_rejected(monkeypatch, tmp_path):
    bridge = FakeBridge(rejections=1)
    install(monkeypatch, bridge)
    submit = lambda *args: ModelOutput.for_tool_call("mock", "submit_action", {"choices": [0]})
    logs = await eval_async(
        position.vgc_position(positions="test"), model=offline_model(submit), log_dir=str(tmp_path)
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    assert bridge.count == 2 and bridge.closed
    assert sample.store["position_choice"] == "a"
    assert [(d["number"], d["error"]) for d in sample.store["decisions"]] == [
        (1, None),
        (2, "rejected"),
    ]


@pytest.mark.parametrize("failure", ["event", "turn", "submit", "provenance"])
async def test_solver_errors_preserve_trace(monkeypatch, tmp_path, failure):
    bridge = FakeBridge(
        unexpected=failure == "event", fail=failure == "submit", turn=3 if failure == "turn" else 2
    )
    if failure == "provenance":
        bridge.hello = {"harness_commit": "wrong"}
    install(monkeypatch, bridge)
    logs = await eval_async(
        position.vgc_position(max_generations=1),
        model=offline_model(),
        log_dir=str(tmp_path),
        fail_on_error=False,
    )
    sample = logs[0].samples[0]
    assert sample.error is not None
    assert bridge.closed
    assert sample.store["completion"] == "incomplete"
    assert sample.store["failure"]["type"] == "BridgeError"
    assert bridge.count == 0
    if failure == "event":
        assert "recorded choice was rejected" in sample.store["failure"]["message"]


@pytest.mark.parametrize("accepted", [True, False])
async def test_scorer_on_demand(monkeypatch, tmp_path, accepted):
    import threading

    bridge = FakeBridge(choice="new")
    install(monkeypatch, bridge)
    main_thread = threading.get_ident()
    calls = []

    def value(data, p, command):
        assert threading.get_ident() != main_thread
        calls.append(command)
        return {"value": 0.9, "explored": 0.8, "label": "new"} if accepted else None

    monkeypatch.setattr(positions, "value_command", value)
    logs = await eval_async(
        position.vgc_position(max_generations=1), model=offline_model(), log_dir=str(tmp_path)
    )
    score = logs[0].samples[0].scores["position_regret"]
    assert calls == ["new"]
    assert score.value["on_demand"] == 1
    assert score.value["scored"] == int(accepted)
    if accepted:
        assert score.value["regret"] == pytest.approx(-0.3)
        assert score.metadata["entry"]["value"] == 0.9
    else:
        assert math.isnan(score.value["regret"])
        assert score.metadata["entry"] is None


def test_task_conditions_hidden_and_validation(monkeypatch):
    data = dataset()
    hidden = copy.deepcopy(data["positions"][0])
    hidden.update(id="hidden", hidden_opponents=1)
    data["positions"].append(hidden)
    monkeypatch.setattr(position, "load_positions", lambda name: data)
    task = position.vgc_position(tool_access="both", limit_hidden=0)
    assert len(task.dataset) == 2
    assert {s.metadata["tool_access"] for s in task.dataset} == {"full", "no_calculators"}
    assert all(s.metadata["game"] == "g-1" for s in task.dataset)
    turn = data["positions"][0]["turn"]
    assert len(position.vgc_position(min_turn=turn).dataset) == 2
    with pytest.raises(ValueError, match="empty"):
        position.vgc_position(min_turn=turn + 1)
    for kwargs in (
        {"max_generations": 0},
        {"limit_hidden": -1},
        {"min_turn": 0},
        {"tool_access": "bad"},
    ):
        with pytest.raises(ValueError):
            position.vgc_position(**kwargs)


async def record_default_game(pool, seed, keep=lambda view: True):
    players = {
        pid: {"name": f"{pid}-default", "team": pool["teams"][i]["packed"], "seat": "external"}
        for i, pid in enumerate(("p1", "p2"))
    }
    bridge = await LeagueBridge.open(pool["format"])
    accepted = 0
    indices = []
    try:
        await bridge.request("start", {"seed": seed, **players})
        while (event := await bridge.next_event())["kind"] != "end":
            pid = event["pid"]
            if event["kind"] == "decision":
                accepted += pid == "p1" and event["row"]["outcome"] == "accepted"
                continue
            view = event["decision"]
            if pid == "p1" and view["phase"] == "turn" and keep(view):
                indices.append(accepted)
            await bridge.request("abandon", {"pid": pid, "exchange": event["exchange"]["id"]})
        outcome = event["outcome"]
    finally:
        await bridge.close()
    assert outcome["error"] is None
    source = {
        "format": pool["format"],
        "seed": seed,
        "names": {pid: p["name"] for pid, p in players.items()},
        "packed": {pid: p["team"] for pid, p in players.items()},
        "choices": {
            pid: [r["action"] for r in outcome["decisions"][pid] if r["outcome"] == "accepted"]
            for pid in players
        },
    }
    return {"id": "real", "source": source, "log": outcome["log"]}, indices


async def resolved_action(bridge, pid):
    while (event := await bridge.next_event())["kind"] != "decision" or event["pid"] != pid:
        assert event["kind"] != "end"
    assert event["row"]["outcome"] == "accepted"
    return event["row"]["action"]


@needs_harness
async def test_real_position_values_and_script():
    recorded, indices = await record_default_game(load_pool("test"), [1, 2, 3, 4])
    index = indices[-1]
    settings = {"samples": 2, "epsilon": 0.25, "maxTurns": 40, "salt": 12}
    rows = positions.value_games(
        [recorded], settings, 1, only={"real": [{"pid": "p1", "choice_index": index}]}
    )
    assert next(r for r in rows if r["kind"] == "game")["verified"]
    (table,) = [r for r in rows if r["kind"] == "position"]
    assert table["turn"] > 1
    command = table["actions"][0]["command"]
    restricted = positions.value_games(
        [recorded],
        settings,
        1,
        only={"real": [{"pid": "p1", "choice_index": index, "commands": [command]}]},
    )
    (single,) = [r for r in restricted if r["kind"] == "position"]
    assert single["actions"] == [table["actions"][0]]
    bridge = await LeagueBridge.open(recorded["source"]["format"])
    try:
        event = await position.resume_position(bridge, recorded["source"], table)
        assert event["decision"]["turn"] == table["turn"]
        await bridge.request("abandon", {"pid": "p1", "exchange": event["exchange"]["id"]})
        assert await resolved_action(bridge, "p1") in {a["command"] for a in table["actions"]}
    finally:
        await bridge.close()
    bridge = await LeagueBridge.open(recorded["source"]["format"])
    try:
        with pytest.raises(BridgeError, match="unexpected first position event"):
            await position.resume_position(bridge, recorded["source"], {**table, "turn": 99})
    finally:
        await bridge.close()


@needs_harness
async def test_real_league_decision_resumes_through_the_task(monkeypatch, tmp_path):
    (hello,) = request_sync([("open", {"format": FORMAT})])
    data = positions.load_positions("league")
    data = {**data, "provenance": hello, "positions": data["positions"][:1]}
    monkeypatch.setattr(position, "load_positions", lambda name: data)
    logs = await eval_async(
        position.vgc_position(max_generations=1), model=offline_model(), log_dir=str(tmp_path)
    )
    sample = logs[0].samples[0]
    assert sample.error is None
    (decision,) = sample.store["decisions"]
    assert (decision["turn"], decision["phase"]) == (data["positions"][0]["turn"], "turn")
    assert decision["defaulted"]
    assert sample.scores["position_regret"].value["scored"] == 1
    assert data["positions"][0]["recorded"]["model"] not in "".join(m.text for m in sample.messages)


def test_build_passes_selection_counts_and_unverified(monkeypatch, tmp_path):
    coarse, choose, select, score = passes()
    games = [game(), {**game(), "id": "bad"}]
    calls = []
    monkeypatch.setattr(positions, "read_run", lambda path: games)
    monkeypatch.setattr(
        positions,
        "request_sync",
        lambda requests: [{"harness_commit": "h", "showdown_commit": "s"}],
    )
    monkeypatch.setattr(positions, "engine_provenance", lambda path: {"source_sha256": "hash"})

    def value(games, settings, jobs, only=None, cache=None):
        calls.append((settings, only))
        if settings["salt"] == 1:
            return [
                {"kind": "game", "id": "g-1", "verified": True},
                {"kind": "game", "id": "bad", "verified": False},
                *coarse,
            ]
        assert only == positions.shortlists(coarse, shortlist=1)
        return {11: choose, 12: select, 13: score}[settings["salt"]]

    monkeypatch.setattr(positions, "value_games", value)
    data = positions.build_dataset(tmp_path, "test", shortlist=1, fine_samples=8, good_margin=0.05)
    assert [s["salt"] for s, _ in calls] == [1, 11, 12, 13]
    assert [s["samples"] for s, _ in calls] == [4, 8, 8, 8]
    assert data["selection"]["unverified_games"] == ["bad"]
    assert data["selection"]["decisions_valued"] == 1
    assert data["selection"]["selected"] == 1
    assert data["selection"]["games"] == 2
    assert data["selection"]["good_margin"] == 0.05
    assert set(data["games"]) == {"g-1"}
    assert data["provenance"]["source_sha256"] == "hash"


def test_verify_recomputes_stored_values_and_restamps_only_when_clean(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(positions, "DATA", tmp_path)
    data = dataset()
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    stored = path.read_text()
    calls = []
    drift = 0.0
    missing = False

    def value(games, settings, jobs, only):
        calls.append((games, settings, jobs, only))
        actions = [
            action(command, entry["value"] + drift, entry["explored"])
            for command, entry in data["positions"][0]["values"].items()
        ]
        table = {"kind": "position", "id": "g-1", "focal": "p1", "choice_index": 2}
        return [{"kind": "game", "id": "g-1", "verified": not missing}] + (
            [] if missing else [{**table, "actions": actions}]
        )

    monkeypatch.setattr(positions, "value_games", value)
    monkeypatch.setattr(
        positions,
        "request_sync",
        lambda requests: [{"harness_commit": "new", "showdown_commit": "s"}],
    )
    monkeypatch.setattr(positions, "engine_provenance", lambda path: {"source_sha256": "hash"})
    monkeypatch.setattr(sys, "argv", ["positions", "verify", "test", "--jobs", "3"])
    positions.main()
    assert "positions: 1; values compared: 4; mismatches: 0" in capsys.readouterr().out
    assert path.read_text() == stored
    assert calls == [
        (
            [game()],
            {"samples": 24, "epsilon": 0.25, "maxTurns": 40, "salt": 13},
            3,
            {
                "g-1": [
                    {"pid": "p1", "choice_index": 2, "commands": ["greedy", "recorded", "a", "b"]}
                ]
            },
        )
    ]
    monkeypatch.setattr(sys, "argv", ["positions", "verify", "test", "--restamp"])
    for drift, missing in ((1e-9, False), (0.0, True)):
        with pytest.raises(SystemExit):
            positions.main()
        assert "mismatches: 4" in capsys.readouterr().out
        assert path.read_text() == stored
    missing = False
    positions.main()
    restamped = json.loads(path.read_text())
    assert restamped.pop("provenance") == {
        "harness_commit": "new",
        "showdown_commit": "s",
        "source_sha256": "hash",
        "revalued": {"from": "h", "values": 4},
    }
    assert restamped == {k: v for k, v in data.items() if k != "provenance"}
    assert list(json.loads(path.read_text())) == list(data)


def test_loader_and_build_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(positions, "DATA", tmp_path)
    for name in ("", "..", "../other", "/tmp/other"):
        with pytest.raises(ValueError):
            positions.load_positions(name)
    with pytest.raises(FileNotFoundError, match="positions build"):
        positions.load_positions("absent")
    path = tmp_path / "test.json"
    path.write_text(json.dumps(dataset()))
    assert positions.load_positions("test")["id"] == "test"
    monkeypatch.setattr(sys, "argv", ["positions", "build", str(tmp_path), "--name", "test"])
    monkeypatch.setattr(
        positions,
        "build_dataset",
        lambda *args, **kwargs: pytest.fail("must refuse before valuation"),
    )
    with pytest.raises(SystemExit):
        positions.main()


async def test_scorer_without_choice_does_not_value(monkeypatch):
    from types import SimpleNamespace

    from inspect_ai.scorer import Target

    from league_evals.scorers import position_regret

    monkeypatch.setattr(
        positions, "value_command", lambda *args: pytest.fail("no command to value")
    )
    state = SimpleNamespace(metadata={"position": dataset()["positions"][0]}, store={})
    score = await position_regret()(state, Target(""))
    assert score.value["scored"] == 0
    assert score.value["on_demand"] == 0
    assert all(
        math.isnan(score.value[k])
        for k in ("value", "regret", "good", "beats_greedy", "matches_recorded")
    )
