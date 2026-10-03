import asyncio
import sys
from types import SimpleNamespace

import pytest

from league_evals.bridge import (
    FORMAT,
    BridgeError,
    BridgeRejected,
    LeagueBridge,
    bridge_command,
    league_dir,
    request_sync,
)

try:
    bridge_command(league_dir())
    HARNESS = True
except FileNotFoundError:
    HARNESS = False

needs_harness = pytest.mark.skipif(not HARNESS, reason="league harness not built; set LEAGUE_DIR")


def players(pool, **seats):
    return {
        pid: {"name": name, "team": pool["teams"][i]["packed"], "seat": seats[pid]}
        for i, (pid, name) in enumerate((("p1", "focal"), ("p2", "opponent")))
    }


@needs_harness
def test_pool_export_reads_packed_teams():
    hello, pool = request_sync([("open", {"format": FORMAT}), ("pool", {"name": "test"})])
    assert len(hello["showdown_commit"]) == 40
    assert {"external", "random", "greedy", "search", "search:fast"} <= set(hello["seats"])
    assert pool["format"].endswith("bo3")
    assert len(pool["teams"]) >= 2
    assert all(team["packed"] for team in pool["teams"])


@needs_harness
async def test_default_policy_completes_a_game_with_audit():
    _, pool = request_sync([("open", {"format": FORMAT}), ("pool", {"name": "test"})])
    bridge = await LeagueBridge.open(pool["format"])
    try:
        with pytest.raises(BridgeRejected, match="external"):
            await bridge.request("start", {"seed": 7, **players(pool, p1="greedy", p2="greedy")})
        await bridge.request(
            "start", {"seed": 7, "policy_seed": 7, **players(pool, p1="external", p2="greedy")}
        )
        rows = []
        exchanges = 0
        while (event := await bridge.next_event())["kind"] != "end":
            assert event["pid"] == "p1"
            if event["kind"] == "decision":
                rows.append(event["row"])
                continue
            exchanges += 1
            exchange, view = event["exchange"], event["decision"]
            assert exchange["task"] == f"decision-{exchanges}"
            assert exchange["submission"]["name"] == "submit_action"
            assert {t["name"] for t in exchange["tools"]} >= {
                "estimate_damage",
                "compare_action_order",
            }
            target = {"pid": "p1", "exchange": exchange["id"]}
            if exchanges != 2:
                await bridge.request("abandon", target)
                continue
            assert view["phase"] == "turn" and len(view["menus"]) == len(view["slot_names"]) == 2
            for refused in (
                {"choices": []},
                {"choices": [0, 0], "bogus": 1},
                {"choices": [999, 0]},
            ):
                with pytest.raises(BridgeRejected):
                    await bridge.request("submit", {**target, "input": refused})
            text = await bridge.request(
                "tool",
                {
                    **target,
                    "name": "compare_action_order",
                    "arguments": {"first": "ally 1", "second": "foe 1"},
                },
            )
            assert "act first" in text
            with pytest.raises(BridgeRejected, match="unknown tool"):
                await bridge.request("tool", {**target, "name": "missing", "arguments": {}})
            submitted = {"choices": [0, 0], "rationale": "checked the order"}
            reply = await bridge.request(
                "submit", {**target, "input": submitted, "usage": {"input_tokens": 10}}
            )
            assert reply == {"accepted": True}
            with pytest.raises(BridgeRejected, match="no pending exchange"):
                await bridge.request("submit", {**target, "input": submitted})
        outcome = event["outcome"]
        assert exchanges > 2
        assert outcome["error"] is None
        assert outcome["winner"] in {"focal", "opponent"}
        assert len(outcome["log_sha256"]) == 64
        assert outcome["decisions"] == {"p1": rows, "p2": []}
        played = [row for row in rows if not row["automatic"]]
        assert [row["submission_source"] for row in played[:3]] == [
            "model-default",
            "model",
            "model-default",
        ]
        assert played[1]["rationale"] == "checked the order"
        assert played[1]["parse_failures"] == 3
        assert played[1]["tool_lookups"] == ["compare_action_order"]
        assert played[1]["total_tokens"] == 10
        assert all(row["outcome"] == "accepted" for row in rows)
        assert await bridge.request("outcome") == outcome
        audit = await bridge.request("audit")
        assert audit["orderPredictions"] == 1
        assert isinstance(audit["findings"], list)
    finally:
        await bridge.close()


async def process_bridge(code, timeout=1):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        code,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return LeagueBridge(process, timeout=timeout, event_timeout=timeout)


@pytest.mark.parametrize(
    "code, message",
    [
        (
            'import sys,json; r=json.loads(input()); print(json.dumps({"id":999,"result":1}))',
            "ID mismatch",
        ),
        ('input(); print("not json")', "failed"),
        ('input(); print("[]")', "not an object"),
        ('import json; input(); print(json.dumps({"id": 1}))', "missing result"),
        ("import sys; input(); sys.exit(1)", "exited"),
        ("import time; input(); time.sleep(30)", "failed"),
    ],
)
async def test_transport_failures_are_bounded_and_process_is_reaped(code, message):
    bridge = await process_bridge(code, timeout=0.5)
    with pytest.raises(BridgeError, match=message):
        await bridge.request("hello")
    assert bridge._process.returncode is not None
    with pytest.raises(BridgeError, match="failed"):
        await bridge.request("hello")
    with pytest.raises(BridgeError):
        await bridge.next_event()
    await bridge.close()


async def test_model_can_retry_rejected_request_on_same_connection():
    bridge = await process_bridge("""import json
for line in __import__('sys').stdin:
 r=json.loads(line)
 print(json.dumps({'id':r['id'], **({'error':'invalid action'} if r['id']==1 else {'result':'ok'})}))
""")
    try:
        with pytest.raises(BridgeRejected):
            await bridge.request("submit")
        assert await bridge.request("submit") == "ok"
    finally:
        await bridge.close()


async def test_events_interleave_with_replies_and_survive_until_the_process_exits():
    bridge = await process_bridge("""import json, sys
def send(message): print(json.dumps(message))
r = json.loads(input())
send({'event': {'kind': 'exchange', 'n': 1}})
send({'id': r['id'], 'result': 'started'})
send({'event': {'kind': 'decision', 'n': 2}})
r = json.loads(input())
send({'id': r['id'], 'result': 'accepted'})
send({'event': {'kind': 'end', 'n': 3}})
print('engine crashed', file=sys.stderr)
sys.exit(1)
""")
    try:
        assert await bridge.request("start") == "started"
        assert await bridge.request("submit") == "accepted"
        assert [(await bridge.next_event())["n"] for _ in range(3)] == [1, 2, 3]
        for _ in range(2):
            with pytest.raises(BridgeError, match="bridge exited: engine crashed"):
                await bridge.next_event()
        with pytest.raises(BridgeError, match="bridge audit failed: bridge exited"):
            await bridge.request("audit")
    finally:
        await bridge.close()


async def test_waiting_for_an_event_is_bounded():
    bridge = await process_bridge("import time; time.sleep(30)", timeout=0.05)
    with pytest.raises(BridgeError, match="no event in time"):
        await bridge.next_event()
    assert bridge._process.returncode is not None


def test_one_shot_requests_skip_event_lines(monkeypatch, tmp_path):
    from league_evals import bridge

    code = """import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(json.dumps({'event': {'kind': 'exchange'}}))
    print(json.dumps({'id': r['id'], 'result': r['method']}))
"""
    monkeypatch.setattr(bridge, "bridge_command", lambda directory: [sys.executable, "-c", code])
    assert request_sync([("open", {}), ("pool", {})], tmp_path) == ["open", "pool"]


@needs_harness
async def test_inspect_completes_both_seats_with_real_engine_and_mock_model(monkeypatch, tmp_path):
    from inspect_ai import eval_async
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from inspect_ai.tool import ToolCall
    from mock_model import offline_model

    from league_evals import battle
    from league_evals.report import collect, comparison

    (hello,) = request_sync([("open", {"format": FORMAT})])
    pool = {**battle.load_pool("test"), "provenance": hello}
    monkeypatch.setattr(battle, "load_pool", lambda name: pool)
    picks = {"choices": [0, 1, 2, 3], "rationale": "lead the first two", "notebook": {}}

    def respond(messages, tools, *args):
        if "Ordered team menu" not in messages[-1].text:
            return ModelOutput.from_content("mock", "No action submitted.")
        calls = [("lookup_move", {"name": "Protect"}), ("submit_action", picks)]
        return ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id=str(i), function=name, arguments=arguments)
                    for i, (name, arguments) in enumerate(calls)
                ],
            )
        )

    logs = await eval_async(
        battle.vgc_battle(max_generations=1, tool_access="both", opponent="greedy"),
        model=offline_model(respond),
        limit=4,
        max_samples=1,
        log_dir=str(tmp_path),
    )
    assert logs[0].status == "success"
    assert {s.metadata["focal_seat"] for s in logs[0].samples} == {"p1", "p2"}
    for sample in logs[0].samples:
        assert sample.error is None
        assert sample.store["completion"] == "complete"
        assert sample.scores["outcome"].value["assisted"] == 1
        assert sample.scores["outcome"].value["unassisted_win"] == 0
        preview, *later = sample.store["decisions"]
        assert (preview["phase"], preview["calls"], preview["defaulted"]) == (
            "team_preview",
            ["lookup_move"],
            False,
        )
        assert later and all(d["defaulted"] for d in later)
        row = sample.store["outcome"]["decisions"][sample.metadata["focal_seat"]][0]
        assert (row["action"], row["submission_source"]) == ("team 1234", "model")
        assert (row["rationale"], row["total_tokens"]) == ("lead the first two", 20)
        names = {tool["name"] for tool in sample.store["tool_catalog"]}
        assert ("estimate_damage" in names) == (sample.metadata["tool_access"] == "full")
        ablated = sample.store["system_prompt"].endswith(battle.NO_CALCULATORS)
        assert ablated == ("estimate_damage" not in names)
        assert len(sample.store["provenance"]["runtime_sha256"]) == 64
    groups = collect(tmp_path, {})
    assert len(groups) == 2
    assert all(row["complete"] for rows in groups.values() for row in rows)
    ids = {rows[0]["config"]["tool_access"]: group for group, rows in groups.items()}
    result = comparison(groups, f"{ids['no_calculators']},{ids['full']}")
    assert result["matched_cells"] == 2
    assert result["missing_outcome_bounds"] == [0, 0]


@needs_harness
async def test_stored_pool_from_another_engine_is_refused_before_generation(tmp_path):
    from inspect_ai import eval_async
    from mock_model import offline_model

    from league_evals import battle

    (hello,) = request_sync([("open", {"format": FORMAT})])
    if battle.load_pool("test")["provenance"]["harness_commit"] == hello["harness_commit"]:
        pytest.skip("the stored pool matches this engine")
    calls = []
    logs = await eval_async(
        battle.vgc_battle(),
        model=offline_model(lambda *args: calls.append(args)),
        limit=1,
        fail_on_error=False,
        log_dir=str(tmp_path),
    )
    assert "pool and engine harness_commit differ" in logs[0].samples[0].error.message
    assert not calls


@needs_harness
async def test_seed_replay_matches_after_removing_wall_clock_timestamps():
    from league_evals.baseline import default_game
    from league_evals.battle import battle_samples, load_pool

    meta = battle_samples(load_pool("test"), [7], "greedy")[1].metadata
    first = await default_game(meta)
    second = await default_game(meta)
    semantic_log = lambda row: [
        line for line in row["outcome"]["log"] if not line.startswith("|t:|")
    ]
    assert first["complete"] and first["decisions"] > 1
    assert first["actions"] == second["actions"]
    assert semantic_log(first) == semantic_log(second)


@needs_harness
async def test_default_policy_finishes_against_the_search_opponent():
    from league_evals.baseline import default_game
    from league_evals.battle import battle_samples, load_pool

    meta = battle_samples(load_pool("test"), [7], "search:fast")[0].metadata
    row = await default_game(meta)
    assert row["complete"] and row["outcome"]["error"] is None
    assert row["scores"]["assisted"] == 1 and row["scores"]["simulator_substitutions"] == 0
    assert row["outcome"]["winner"] in {"focal", "opponent"}
    with pytest.raises(BridgeError, match="opponent must be one of random, greedy, search"):
        await default_game({**meta, "opponent_policy": "external"})


class ExitsBeforeKill:
    returncode = None

    def __init__(self) -> None:
        self.stdin = SimpleNamespace(close=lambda: None)
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()
        self.waits = 0

    async def wait(self) -> int:
        self.waits += 1
        if self.waits == 1:
            raise TimeoutError
        return 0

    def kill(self) -> None:
        raise ProcessLookupError


async def test_close_tolerates_a_process_that_exits_before_kill():
    process = ExitsBeforeKill()
    await LeagueBridge(process).close()
    assert process.waits == 2
