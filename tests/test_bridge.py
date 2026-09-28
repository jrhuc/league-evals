import asyncio
from types import SimpleNamespace

import pytest

from league_evals.bridge import BridgeError, LeagueBridge, bridge_command, league_dir, request_sync

try:
    bridge_command(league_dir())
    HARNESS = True
except FileNotFoundError:
    HARNESS = False

needs_harness = pytest.mark.skipif(not HARNESS, reason="league harness not built; set LEAGUE_DIR")


@needs_harness
def test_pool_export_reads_packed_teams():
    hello, pool = request_sync([("open", {}), ("pool", {"name": "test"})])
    assert len(hello["showdown_commit"]) == 40
    assert pool["format"].endswith("bo3")
    assert len(pool["teams"]) >= 2
    assert all(team["packed"] for team in pool["teams"])


@needs_harness
async def test_default_policy_completes_a_game_with_audit():
    _, pool = request_sync([("open", {}), ("pool", {"name": "test"})])
    bridge = await LeagueBridge.open(pool["format"])
    try:
        tools = await bridge.request("tools")
        assert {t["name"] for t in tools} >= {"estimate_damage", "compare_action_order"}
        event = await bridge.request(
            "start",
            {
                "format": pool["format"],
                "seed": 7,
                "p1": {"name": "focal", "team": pool["teams"][0]["packed"]},
                "p2": {"name": "opponent", "team": pool["teams"][1]["packed"]},
                "external": ["p1"],
                "opponent_seed": 7,
            },
        )
        decisions = 0
        while event["kind"] == "decision":
            decisions += 1
            if event["phase"] == "turn" and decisions == 2:
                with pytest.raises(BridgeError):
                    await bridge.request("submit", {"pid": "p1", "choices": []})
                text = await bridge.request(
                    "call",
                    {
                        "pid": "p1",
                        "name": "compare_action_order",
                        "arguments": {"first": "ally 1", "second": "foe 1"},
                    },
                )
                assert text
            reply = await bridge.request("submit", {"pid": "p1", "choices": "default"})
            event = reply["next"]
        assert decisions > 1
        outcome = event["outcome"]
        assert outcome["winner"] in {"focal", "opponent"}
        assert len(outcome["log_sha256"]) == 64
        audit = await bridge.request("audit")
        assert isinstance(audit["findings"], list)
    finally:
        await bridge.close()


async def process_bridge(code, timeout=1):
    import asyncio
    import sys

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        code,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return LeagueBridge(process, timeout=timeout)


@pytest.mark.parametrize(
    "code, message",
    [
        (
            'import sys,json; r=json.loads(input()); print(json.dumps({"id":999,"result":1}))',
            "ID mismatch",
        ),
        ('input(); print("not json")', "failed"),
        ("import sys; input(); sys.exit(1)", "exited"),
        ("import time; input(); time.sleep(30)", "failed"),
    ],
)
async def test_transport_failures_are_bounded_and_process_is_reaped(code, message):
    bridge = await process_bridge(code, timeout=0.05)
    with pytest.raises(BridgeError, match=message):
        await bridge.request("hello")
    assert bridge._process.returncode is not None
    await bridge.close()


async def test_model_can_retry_rejected_request_on_same_connection():
    from league_evals.bridge import BridgeRejected

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


@needs_harness
async def test_inspect_completes_both_seats_with_real_engine_and_mock_model(tmp_path):
    from inspect_ai import eval_async
    from mock_model import offline_model

    from league_evals.battle import vgc_battle
    from league_evals.report import collect, comparison

    logs = await eval_async(
        vgc_battle(max_generations=1, tool_access="both"),
        model=offline_model(),
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
        assert sample.store["decisions"]
        names = {tool["name"] for tool in sample.store["tool_catalog"]}
        assert ("estimate_damage" in names) == (sample.metadata["tool_access"] == "full")
        assert len(sample.store["provenance"]["runtime_sha256"]) == 64
    groups = collect(tmp_path, {})
    assert len(groups) == 2
    assert all(row["complete"] for rows in groups.values() for row in rows)
    ids = {rows[0]["config"]["tool_access"]: group for group, rows in groups.items()}
    result = comparison(groups, f"{ids['no_calculators']},{ids['full']}")
    assert result["matched_cells"] == 2
    assert result["missing_outcome_bounds"] == [0, 0]


@needs_harness
async def test_seed_replay_matches_after_removing_wall_clock_timestamps():
    from league_evals.baseline import default_game
    from league_evals.battle import battle_samples, load_pool

    meta = battle_samples(load_pool("test"), [7])[1].metadata
    first = await default_game(meta)
    second = await default_game(meta)
    semantic_log = lambda row: [
        line for line in row["outcome"]["log"] if not line.startswith("|t:|")
    ]
    assert first["actions"] == second["actions"]
    assert semantic_log(first) == semantic_log(second)


class ExitsBeforeKill:
    returncode = None

    def __init__(self) -> None:
        self.stdin = SimpleNamespace(close=lambda: None)
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
