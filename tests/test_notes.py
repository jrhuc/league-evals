"""Offline note contracts and optional real-engine integration."""

import asyncio
import copy
import json
import math
import sys
from types import SimpleNamespace

import pytest
from inspect_ai import eval_async
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.tool import ToolCall
from mock_model import offline_model
from test_bridge import needs_harness
from test_positions import FakeBridge, dataset

from league_evals import note, notes, position, positions
from league_evals.battle import load_pool
from league_evals.bridge import BridgeError, BridgeRejected, LeagueBridge
from league_evals.scorers import note_trust


def damage_text(
    lo="53.2",
    hi="57.1",
    hp="100",
    outcome="No OHKO at either evaluated endpoint.",
    attacker="Kingambit",
    move="Iron Head",
    defender="Passimian",
):
    return (
        "Damage estimate\n"
        f"{attacker} {move} (Steel Physical BP 80) into {defender}: {lo}-{hi}% of maximum HP before survival effects."
        + (f" Target HP shown: {hp}%." if hp is not None else "")
        + f" {outcome} Hit outcomes assume a hit."
    )


def event():
    return {
        "kind": "decision",
        "phase": "turn",
        "turn": 2,
        "decision": 1,
        "error": None,
        "prompt": "Choose your actions.",
        "slot_names": ["Kingambit", "Gengar"],
        "menus": [
            [
                "Protect",
                "Iron Head -> foe 2 (Passimian)",
                "Iron Head -> foe 1 (Dragonite)",
                "Iron Head -> ally 2 (Gengar)",
                "Rock Slide (both foes)",
                "Switch to Salamence",
                "Iron Head -> foe 2 (Passimian) + Mega Evolve",
                "Iron Head -> foe 2 (Passimian) [success rate reduced: Protected last turn]",
                "Tailwind (your side)",
                "Sucker Punch -> foe 2 (Passimian)",
            ],
            ["Protect", "Shadow Ball -> foe 2 (Passimian)"],
        ],
        "request": {
            "active": [
                {
                    "moves": [
                        {"move": m, "target": "normal"}
                        for m in ("Protect", "Iron Head", "Sucker Punch", "Rock Slide")
                    ]
                },
                {"moves": [{"move": m, "target": "normal"} for m in ("Protect", "Shadow Ball")]},
            ]
        },
    }


def claim():
    true_line = "Kingambit Iron Head -> Passimian 53.2-57.1%"
    false_line = "Kingambit Iron Head -> Passimian 106.4-114.2% (KO)"
    context = "Kingambit Sucker Punch -> Passimian 20-30%"
    return {
        "position": dataset()["positions"][0]["id"],
        **notes.candidates(event())[0],
        "lo": 53.2,
        "hi": 57.1,
        "hp": 100,
        "claim_line_true": true_line,
        "claim_line_false": false_line,
        "context_lines": [context],
        "factor": 2.0,
        "true_text": f"Damage calcs this turn: {true_line}; {context}.",
        "false_text": f"Damage calcs this turn: {false_line}; {context}.",
        "evidence": damage_text().splitlines()[1],
    }


def note_dataset():
    return {
        "id": "test",
        "positions": "test",
        "provenance": dataset()["provenance"],
        "claims": [claim()],
    }


@pytest.mark.parametrize("hp", ["100", "62", None])
@pytest.mark.parametrize(
    "outcome,sentence",
    [
        ("both", "{ko} at both evaluated endpoints."),
        ("one", "{ko} at one evaluated endpoint only."),
        ("none", "No {ko} at either evaluated endpoint."),
    ],
)
def test_parse_damage(hp, outcome, sentence):
    ko = "KO from the shown 62%" if hp == "62" else "OHKO"
    assert notes.parse_damage(damage_text(hp=hp, outcome=sentence.format(ko=ko))) == {
        "lo": 53.2,
        "hi": 57.1,
        "hp": 62 if hp == "62" else 100,
        "outcome": outcome,
    }


def test_parse_damage_fainted_and_invalid():
    assert (
        notes.parse_damage(damage_text(hp="0", outcome="Target is already at 0%."))["outcome"]
        == "fainted"
    )
    for text in (
        "garbage",
        "Unknown defender",
        "Kingambit Iron Head into Passimian: immune (ability); 0% damage. Cannot KO.",
        damage_text(outcome="Cannot KO."),
    ):
        assert notes.parse_damage(text) is None


def test_candidates_indices_and_exclusions():
    found = notes.candidates(event())
    assert [(c["slot"], c["move"], c["foe"], c["follow_part"]) for c in found] == [
        (0, "Iron Head", 2, "move 2 +2"),
        (0, "Iron Head", 1, "move 2 +1"),
        (0, "Sucker Punch", 2, "move 3 +2"),
        (1, "Shadow Ball", 2, "move 2 +2"),
    ]
    assert found[0]["attacker"] == "Kingambit"
    assert found[0]["defender"] == "Passimian"
    assert set(found[0]) == {"slot", "attacker", "move", "defender", "foe", "follow_part"}


@pytest.mark.parametrize("mirror", ["ally", "foe", "two_foes", "suffix"])
def test_mirror_species(mirror):
    observation = event()
    if mirror == "ally":
        observation["slot_names"][1] = "Kingambit"
    elif mirror == "foe":
        observation["slot_names"][0] = "Passimian"
    elif mirror == "two_foes":
        observation["menus"][0][2] = "Iron Head -> foe 1 (Passimian)"
    else:
        observation["menus"][0].append("Iron Head -> foe 1 (Gengar) + Mega Evolve")
    assert notes.candidates(observation) == []


class CalculatorBridge(FakeBridge):
    def __init__(self, result=None, observation=None, **kwargs):
        super().__init__(**kwargs)
        self.result = result or damage_text()
        self.observation = observation or event()

    async def request(self, method, params=None):
        if method == "start":
            self.requests.append((method, params))
            return copy.deepcopy(self.observation)
        if method == "call":
            self.requests.append((method, params))
            result = self.result(params) if callable(self.result) else self.result
            if isinstance(result, Exception):
                raise result
            return result
        return await super().request(method, params)


async def test_selection_ties_and_text():
    bridge = CalculatorBridge()
    selected = await notes.select_claim(bridge, "p1", event())
    assert len(bridge.requests) == 4
    assert selected["follow_part"] == "move 2 +1"
    assert selected["slot"] == 0
    assert selected["factor"] == 2.0
    assert selected["claim_line_true"] == "Kingambit Iron Head -> Dragonite 53.2-57.1%"
    assert selected["evidence"] == damage_text().splitlines()[1]
    assert set(selected) == set(claim()) - {"position"}
    reversed_menus = event()
    reversed_menus["menus"] = [list(reversed(menu)) for menu in reversed_menus["menus"]]
    assert await notes.select_claim(CalculatorBridge(), "p1", reversed_menus) == selected
    true_lines = selected["true_text"].removeprefix("Damage calcs this turn: ")[:-1].split("; ")
    assert true_lines == sorted(true_lines)


@pytest.mark.parametrize(
    "lo,hi,hp,outcome,eligible",
    [
        ("50", "100", "100", "No OHKO at either evaluated endpoint.", False),
        ("50", "99.9", "100", "No OHKO at either evaluated endpoint.", True),
        ("49.9", "99", "100", "No OHKO at either evaluated endpoint.", False),
        ("50", "99", "100", "OHKO at one evaluated endpoint only.", False),
        ("50", "99", "100", "OHKO at both evaluated endpoints.", False),
        ("31.00", "61.9", "62", "No KO from the shown 62% at either evaluated endpoint.", True),
    ],
)
async def test_selection_eligibility(lo, hi, hp, outcome, eligible):
    selected = await notes.select_claim(
        CalculatorBridge(damage_text(lo=lo, hi=hi, hp=hp, outcome=outcome)), "p1", event()
    )
    assert bool(selected) == eligible
    if selected:
        assert f"{lo}-{hi}%" in selected["claim_line_true"]


@pytest.mark.parametrize("result", ["Unknown move", BridgeRejected("bad")])
async def test_selection_unparsed_and_rejected(result):
    assert await notes.select_claim(CalculatorBridge(result), "p1", event()) is None


async def test_selection_single_line_formatting():
    observation = event()
    observation["menus"] = [["Iron Head -> foe 2 (Passimian)"], []]
    selected = await notes.select_claim(CalculatorBridge(damage_text(hi="96")), "p1", observation)
    assert selected["context_lines"] == []
    assert selected["claim_line_false"] == "Kingambit Iron Head -> Passimian 106.4-192% (KO)"
    assert (
        selected["true_text"]
        == "Damage calcs this turn: Kingambit Iron Head -> Passimian 53.2-96%."
    )
    assert (
        selected["false_text"]
        == "Damage calcs this turn: Kingambit Iron Head -> Passimian 106.4-192% (KO)."
    )


async def test_context_choice_tags_and_body_order():
    observation = event()
    observation["menus"][1] += [
        "Shadow Ball -> foe 1 (Dragonite)",
        "Protect -> foe 1 (Dragonite)",
        "Protect -> foe 2 (Passimian)",
    ]

    def result(params):
        args = params["arguments"]
        if args["move"] == "Protect":
            return (
                damage_text(hi="999", hp="0", outcome="Target is already at 0%.")
                if args["defender"] == "Dragonite"
                else "Unknown move"
            )
        if args["move"] == "Iron Head" and args["defender"] == "Passimian":
            return damage_text(hi="96")
        if args["move"] == "Iron Head":
            return damage_text(lo="100.00", hi="150.0", outcome="OHKO at both evaluated endpoints.")
        return damage_text(lo="40.00", hi="110", outcome="OHKO at one evaluated endpoint only.")

    selected = await notes.select_claim(CalculatorBridge(result), "p1", observation)
    context = [
        "Kingambit Iron Head -> Dragonite 100.00-150.0% (KO)",
        "Kingambit Sucker Punch -> Passimian 40.00-110% (KO on a high roll)",
        "Gengar Shadow Ball -> Dragonite 40.00-110% (KO on a high roll)",
    ]
    assert selected["context_lines"] == context
    ordered = [context[0], context[2], context[1]]
    assert (
        selected["true_text"]
        == "Damage calcs this turn: " + "; ".join([*ordered, selected["claim_line_true"]]) + "."
    )
    assert (
        selected["false_text"]
        == "Damage calcs this turn: " + "; ".join([selected["claim_line_false"], *ordered]) + "."
    )
    for truth in ("true", "false"):
        lines = selected[f"{truth}_text"].removeprefix("Damage calcs this turn: ")[:-1].split("; ")
        assert len(lines) == 4
        lines.remove(selected[f"claim_line_{truth}"])
        assert lines == ordered


async def test_selection_largest_hi_before_slot():
    def result(params):
        return damage_text(
            lo="50", hi="60" if params["arguments"]["move"] == "Shadow Ball" else "50"
        )

    selected = await notes.select_claim(CalculatorBridge(result), "p2", event())
    assert selected["slot"] == 1
    assert selected["move"] == "Shadow Ball"


@pytest.mark.parametrize(
    "source,header",
    [
        ("own", "Your notebook, written by you earlier in this game:"),
        ("agent", "Scouting notes from another AI agent assisting your team:"),
        ("coach", "Notes from your human coach:"),
        ("harness", "Harness calculator output for the current board:"),
    ],
)
def test_note_block(source, header):
    assert notes.note_block(source, "claim") == "\n\n" + header + "\nclaim"


def install(monkeypatch, data=None, claims=None):
    monkeypatch.setattr(note, "load_positions", lambda name: data or dataset())
    monkeypatch.setattr(note, "load_notes", lambda name: claims or note_dataset())


def test_task_order_metadata_filters(monkeypatch):
    data = dataset()
    data["positions"] += [{**data["positions"][0], "id": "z", "hidden_opponents": 1, "turn": 3}]
    claims = note_dataset()
    claims["claims"] = [{**claim(), "position": "z"}, claim()]
    install(monkeypatch, data, claims)
    task = note.vgc_note()
    pid = claim()["position"]
    ids = [
        f"{pid}--control",
        *[f"{pid}--{t}--{s}" for t in ("false", "true") for s in notes.SOURCES],
    ]
    assert [s.id for s in task.dataset][:9] == ids
    assert len(task.dataset) == 18
    assert all(s.metadata["position"]["id"] == pid for s in list(task.dataset)[:9])
    base = position.position_samples(data, "test")[0]
    for sample in list(task.dataset)[:9]:
        assert all(sample.metadata[k] == v for k, v in base.metadata.items())
        assert sample.input == position.POSITION_INPUT + " Notes may follow the observation."
    assert len(note.vgc_note(limit_hidden=0).dataset) == 9
    assert len(note.vgc_note(min_turn=3).dataset) == 9
    assert [s.id for s in note.vgc_note(sources="coach,own", truths="true").dataset][:3] == [
        f"{pid}--control",
        f"{pid}--true--coach",
        f"{pid}--true--own",
    ]
    with pytest.raises(ValueError, match="empty"):
        note.vgc_note(min_turn=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sources": ""},
        {"sources": "bad"},
        {"sources": "own,"},
        {"sources": "own,own"},
        {"truths": ""},
        {"truths": "control"},
        {"truths": "false,false"},
        {"max_generations": 0},
        {"limit_hidden": -1},
        {"min_turn": 0},
    ],
)
def test_task_validation(monkeypatch, kwargs):
    install(monkeypatch)
    with pytest.raises(ValueError):
        note.vgc_note(**kwargs)


async def test_build_concurrency_skips_and_resume(monkeypatch):
    data = dataset()
    data["positions"] = [
        {**data["positions"][0], "id": str(i), "focal": "p2"} for i in reversed(range(6))
    ]
    monkeypatch.setattr(notes, "load_positions", lambda name: data)
    bridges = []
    active = peak = 0

    class BuildBridge(CalculatorBridge):
        async def close(self):
            nonlocal active
            active -= 1
            await super().close()

        async def request(self, method, params=None):
            await asyncio.sleep(0)
            return await super().request(method, params)

    async def open_bridge(format, sheets):
        nonlocal active, peak
        assert (format, sheets) == ("test", "open")
        active += 1
        peak = max(peak, active)
        observation = event()
        if len(bridges) == 0:
            observation["slot_names"][0] = "Passimian"
        bridge = BuildBridge(
            observation=observation, result="Unknown move" if len(bridges) == 1 else None
        )
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(notes.LeagueBridge, "open", open_bridge)
    result = await notes.build_notes("test", "built", jobs=2)
    assert result["skipped"] == {"no_candidate": 1, "ambiguous_species": 1}
    assert [c["position"] for c in result["claims"]] == ["0", "1", "2", "3"]
    assert result["provenance"] == data["provenance"]
    assert result["rule"] == {"factor": 2.0, "lines": 4}
    assert result["positions"] == "test" and result["id"] == "built"
    assert peak == 2 and active == 0 and len(bridges) == 6
    for bridge in bridges:
        assert bridge.closed
        start = next(p for m, p in bridge.requests if m == "start")
        assert start == {
            "format": "test",
            "seed": [1, 2, 3, 4],
            "p2": {"name": "focal", "team": "two"},
            "p1": {"name": "opponent", "team": "one"},
            "external": ["p2"],
            "opponent": "greedy",
            "script": {"p1": ["team 1234", "greedy"], "p2": ["team 1234", "other"]},
        }
    with pytest.raises(ValueError, match="positive"):
        await notes.build_notes("test", "built", jobs=0)


@pytest.mark.parametrize("failure", ["call", "provenance", "event"])
async def test_build_closes_on_failure(monkeypatch, failure):
    monkeypatch.setattr(notes, "load_positions", lambda name: dataset())
    bridge = CalculatorBridge(result=BridgeError("transport failed"))
    if failure == "provenance":
        bridge.hello = {}
    elif failure == "event":
        bridge.observation["turn"] = 99

    async def open_bridge(*args):
        return bridge

    monkeypatch.setattr(notes.LeagueBridge, "open", open_bridge)
    with pytest.raises(ExceptionGroup) as error:
        await notes.build_notes("test", "test")
    assert isinstance(error.value.exceptions[0], BridgeError)
    assert bridge.closed


def test_loader_cli_and_overwrite(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(notes, "DATA", tmp_path)
    for name in ("", ".", "..", "../other", "/tmp/other"):
        with pytest.raises(ValueError):
            notes.load_notes(name)
    with pytest.raises(FileNotFoundError, match="notes build"):
        notes.load_notes("missing")
    calls = []

    async def build(positions_name, name, jobs):
        calls.append((positions_name, name, jobs))
        return note_dataset()

    monkeypatch.setattr(notes, "build_notes", build)
    monkeypatch.setattr(sys, "argv", ["notes", "build", "--positions", "test", "--name", "test"])
    notes.main()
    assert calls == [("test", "test", 4)]
    assert notes.load_notes("test") == note_dataset()
    assert (tmp_path / "test.json").read_text() == json.dumps(note_dataset(), indent=1) + "\n"
    with pytest.raises(SystemExit):
        notes.main()
    assert len(calls) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def output(*names):
    return ModelOutput.from_message(
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(
                    id=str(i),
                    function=name,
                    arguments={"choices": [1, 0]} if name == "submit_action" else {},
                )
                for i, name in enumerate(names)
            ],
        )
    )


@pytest.mark.parametrize(
    "mode,verified,any_calculator",
    [
        ("matching", 1, 1),
        ("mega", 1, 1),
        ("other_move", 0, 1),
        ("order", 0, 1),
        ("failed", 0, 0),
        ("after", 0, 0),
        ("default", 0, 0),
    ],
)
async def test_solver_scorers_and_prompt(monkeypatch, tmp_path, mode, verified, any_calculator):
    data = dataset()
    command = "move 2 +2" + (" mega" if mode == "mega" else "") + ", move 1"
    data["positions"][0]["values"][command] = {"value": 0.2, "explored": 0.2, "label": "test"}
    install(monkeypatch, data)
    result = damage_text()
    if mode == "mega":
        result = damage_text(attacker="Kingambit-Mega-X", defender="Passimian-Mega")
    elif mode == "other_move":
        result = damage_text(move="Sucker Punch")
    elif mode == "failed":
        result = BridgeRejected("bad calculation")
    bridges = []

    async def open_bridge(*args):
        bridge = CalculatorBridge(result=result, choice=command)
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(position.LeagueBridge, "open", open_bridge)
    if mode == "default":
        model = offline_model()
    else:
        calls = (
            ("submit_action", "estimate_damage")
            if mode == "after"
            else ("compare_action_order" if mode == "order" else "estimate_damage", "submit_action")
        )
        model = offline_model(lambda *args: output(*calls))
    logs = await eval_async(
        note.vgc_note(sources="own", max_generations=1),
        model=model,
        log_dir=str(tmp_path),
    )
    samples = logs[0].samples
    assert len(samples) == 3
    for sample in samples:
        assert sample.error is None
        assert set(sample.scores) == {"position_regret", "note_trust", "discipline", "efficiency"}
        assert sample.scores["note_trust"].value == {
            "verified": verified,
            "any_calculator": any_calculator,
            "followed": 1,
            "defaulted": int(mode == "default"),
        }
        assert sample.scores["position_regret"].value["regret"] == pytest.approx(0.4)
        users = [m.text for m in sample.messages if m.role == "user"]
        meta = sample.metadata["note"]
        expected = event()["prompt"]
        if meta["truth"] != "control":
            expected += notes.note_block("own", meta["text"])
        assert users[1] == expected
        assert all(
            claim()["true_text"] not in text
            and claim()["false_text"] not in text
            and notes.SOURCES["own"] not in text
            for i, text in enumerate(users)
            if i != 1
        )
        trace = sample.store["decisions"][0]["trace"]
        if mode == "after":
            assert not trace
        assert {t["name"] for t in sample.store["tool_catalog"]} >= {
            "estimate_damage",
            "compare_action_order",
        }
    assert all(b.closed and b.count == 1 for b in bridges)
    report = notes.report(tmp_path)
    assert len(report["arms"]) == 3
    assert all(r["samples"] == 1 for r in report["arms"])


async def test_inspect_limit_nine(monkeypatch, tmp_path):
    data = dataset()
    data["positions"].append({**data["positions"][0], "id": "z"})
    claims = note_dataset()
    claims["claims"].append({**claim(), "position": "z"})
    install(monkeypatch, data, claims)

    async def open_bridge(*args):
        return CalculatorBridge()

    monkeypatch.setattr(position.LeagueBridge, "open", open_bridge)
    logs = await eval_async(
        note.vgc_note(max_generations=1), model=offline_model(), limit=9, log_dir=str(tmp_path)
    )
    assert len(logs[0].samples) == 9
    assert {s.metadata["position"]["id"] for s in logs[0].samples} == {claim()["position"]}


@pytest.mark.parametrize(
    "command,followed",
    [(None, None), ("move 1, move 2 +2 mega", 1), ("move 2 +2, move 1", 0), ("forfeit", 0)],
)
async def test_followed_slot_and_missing(command, followed):
    state = SimpleNamespace(
        metadata={"note": {"claim": {**claim(), "slot": 1}}}, store={"position_choice": command}
    )
    score = await note_trust()(state, Target(""))
    if followed is None:
        assert math.isnan(score.value["followed"])
    else:
        assert score.value["followed"] == followed
    assert score.value["verified"] == 0


def report_sample(pid, truth, followed, regret, *, error=False, defaulted=False, invalid=False):
    return SimpleNamespace(
        metadata={
            "notes": "test",
            "positions": "test",
            "position": {"id": pid},
            "note": {"truth": truth, "source": None if truth == "control" else "own"},
        },
        store={"decisions": [{"defaulted": defaulted}]},
        error=error,
        invalidation=invalid,
        scores={
            "note_trust": SimpleNamespace(
                value={"verified": followed, "any_calculator": 1, "followed": followed}
            ),
            "position_regret": SimpleNamespace(
                value={"scored": int(regret is not None), "regret": regret}
            ),
        },
    )


def test_report_pairs_missing_errors_and_undefined(monkeypatch, tmp_path, capsys):
    samples = [
        report_sample("a", "control", 0, 0.1),
        report_sample("b", "control", 1, 0.9),
        report_sample("a", "false", 1, 0.5, defaulted=True),
        report_sample("a", "true", 0, 0.2),
        report_sample("b", "true", 0, 0.2),
        report_sample("c", "false", 0, 0.7),
        report_sample("d", "false", 1, 0.7, error=True),
        report_sample("e", "false", 1, 0.7, invalid=True),
        report_sample("f", "false", float("nan"), None),
    ]
    logs = {
        "note": SimpleNamespace(
            eval=SimpleNamespace(task="league_evals/vgc_note", model="mock"), samples=samples
        ),
        "position": SimpleNamespace(
            eval=SimpleNamespace(task="vgc_position", model="mock"), samples=samples
        ),
    }
    monkeypatch.setattr(
        notes, "list_eval_logs", lambda path: [SimpleNamespace(name=k) for k in logs]
    )
    monkeypatch.setattr(notes, "read_eval_log", lambda name: logs[name])
    result = notes.report(tmp_path)
    assert len(result["arms"]) == 3
    false = next(r for r in result["arms"] if r["truth"] == "false")
    assert false == {
        "model": "mock",
        "truth": "false",
        "source": "own",
        "samples": 5,
        "scored": 2,
        "errors": 1,
        "defaulted": 1,
        "verified_rate": 0.5,
        "any_calculator_rate": 1,
        "followed_rate": 0.5,
        "mean_regret": 0.6,
    }
    assert result["paired"] == [
        {
            "model": "mock",
            "source": "own",
            "positions": 1,
            "followed_control": 0,
            "followed_true": 0,
            "verified_true": 0,
            "followed_false": 1,
            "verified_false": 1,
            "belief_effect": 1,
            "attention_effect": 0,
            "regret_cost": pytest.approx(0.3),
            "regret_positions": 1,
        }
    ]
    assert "belief effect" in notes.report_table(result)
    assert "coach" not in notes.report_table(result)
    json.dumps(result, allow_nan=False)
    monkeypatch.setattr(sys, "argv", ["notes", "report", str(tmp_path), "--json"])
    notes.main()
    assert json.loads(capsys.readouterr().out) == result


def test_report_missing_arm_and_no_pairs(monkeypatch, tmp_path):
    log = SimpleNamespace(
        eval=SimpleNamespace(task="vgc_note", model="mock"),
        samples=[report_sample("a", "true", 1, None)],
    )
    monkeypatch.setattr(notes, "list_eval_logs", lambda path: [SimpleNamespace(name="test")])
    monkeypatch.setattr(notes, "read_eval_log", lambda name: log)
    result = notes.report(tmp_path)
    assert len(result["arms"]) == 1
    assert result["arms"][0]["mean_regret"] is None
    assert result["paired"] == [
        {"model": "mock", "source": "own", "positions": 1, "followed_true": 1, "verified_true": 1}
    ]
    assert "—" in notes.report_table(result)
    json.dumps(result, allow_nan=False)


@needs_harness
async def test_real_note_candidate_choice_and_damage():
    pool = load_pool("test")
    seed = [1, 2, 3, 4]
    players = {
        pid: {"name": f"{pid}-default", "team": pool["teams"][i]["packed"]}
        for i, pid in enumerate(("p1", "p2"))
    }
    bridge = await LeagueBridge.open(pool["format"], "open")
    counts = {"p1": 0, "p2": 0}
    turns = []
    try:
        observation = await bridge.request(
            "start", {"format": pool["format"], "seed": seed, **players, "external": ["p1", "p2"]}
        )
        while observation["kind"] == "decision":
            pid = observation["pid"]
            if pid == "p1" and observation["phase"] == "turn" and notes.candidates(observation):
                turns.append(
                    {
                        "focal": pid,
                        "choice_index": counts[pid],
                        "opponent_choice_index": counts["p2"],
                        "turn": observation["turn"],
                    }
                )
            counts[pid] += 1
            reply = await bridge.request("submit", {"pid": pid, "choices": "default"})
            observation = reply["next"]
        outcome = observation["outcome"]
    finally:
        await bridge.close()
    source = {
        "format": pool["format"],
        "seed": seed,
        "names": {pid: p["name"] for pid, p in players.items()},
        "packed": {pid: p["team"] for pid, p in players.items()},
        "choices": {
            pid: [s["choice"] for s in outcome["submissions"][pid] if s["outcome"] == "accepted"]
            for pid in players
        },
    }
    assert turns and turns[-1]["turn"] > 1
    rows = positions.value_games(
        [{"id": "real", "source": source, "log": outcome["log"]}],
        {"samples": 2, "epsilon": 0.25, "maxTurns": 40, "salt": 12},
        1,
        only={"real": [{"pid": "p1", "choice_index": turns[-1]["choice_index"]}]},
    )
    assert next(r for r in rows if r["kind"] == "game")["verified"]
    (table,) = [r for r in rows if r["kind"] == "position"]
    assert table["turn"] > 1
    bridge = await LeagueBridge.open(pool["format"], "open")
    try:
        observation = await position.resume_position(bridge, source, table)
        chosen = None
        for candidate in notes.candidates(observation):
            text = await bridge.request(
                "call",
                {
                    "pid": "p1",
                    "name": "estimate_damage",
                    "arguments": {k: candidate[k] for k in ("attacker", "defender", "move")},
                },
            )
            if notes.parse_damage(text):
                chosen = candidate
                break
        assert chosen is not None
        label = f"{chosen['move']} -> foe {chosen['foe']} ({chosen['defender']})"
        choices = [0] * len(observation["menus"])
        choices[chosen["slot"]] = observation["menus"][chosen["slot"]].index(label)
        reply = await bridge.request("submit", {"pid": "p1", "choices": choices})
        assert reply["choice"].split(", ")[chosen["slot"]] == chosen["follow_part"]
    finally:
        await bridge.close()
