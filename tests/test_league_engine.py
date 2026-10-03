import json
import re
from pathlib import Path

from inspect_ai import Task, eval_async
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from mock_model import offline_model
from test_bridge import needs_harness

from league_evals import league
from league_evals.scorers import league_conduct, league_standing

PROBE_SETS = json.loads((Path(__file__).parent / "league_probe_sets.json").read_text())
SEASON_REVIEW = {"summary": "-", "did_well": "-", "did_poorly": "-", "would_change": "-"}
ANSWERS = {
    "submit_name": [{"team_name": "Probe"}],
    "submit_action": [{"choices": [0, 0]}, {"choices": [0]}],
    "submit_review": [SEASON_REVIEW, {"summary": "Fine."}, {}],
    "submit_offer": [{"offer": None}],
    "submit_response": [{"accept": False}],
    "submit_free_agency": [{"swaps": []}],
}


def call(name, arguments):
    return ModelOutput.for_tool_call("mock", name, arguments)


class Probe:
    """A scripted manager that drafts players it holds legal sets for and answers every other task
    with the simplest input the harness accepts."""

    def __init__(self):
        self.owned: list[str] = []

    def __call__(self, messages, tools, *args):
        submission = next(tool.name for tool in tools if tool.name.startswith("submit_"))
        start = max(i for i, message in enumerate(messages) if message.role == "user")
        recent = messages[start:]
        failures = sum(1 for m in recent if m.role == "tool" and m.error is not None)
        self.track(messages)
        if submission == "submit_pick":
            return self.pick(recent, failures)
        if submission == "submit_team":
            return call("submit_team", {"sets": self.sets(), "team_plan": "Probe"})
        answers = ANSWERS[submission]
        if failures >= len(answers):
            return ModelOutput.from_content("mock", "No answer fits.")
        return call(submission, answers[failures])

    def track(self, messages):
        results = {m.tool_call_id: m for m in messages if m.role == "tool"}
        for message in messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                result = results.get(tool_call.id)
                pick = tool_call.arguments.get("pick")
                accepted = result is not None and result.error is None
                if tool_call.function == "submit_pick" and accepted and pick not in self.owned:
                    self.owned.append(pick)

    def pick(self, recent, failures):
        wanted = [mon for mon in PROBE_SETS if mon not in self.owned]
        if failures < len(wanted):
            return call("submit_pick", {"pick": wanted[failures]})
        listed = [m.text for m in recent if m.role == "tool" and m.function == "search_board"]
        if not listed:
            return call("search_board", {})
        first = re.search(r"^- ([a-z0-9-]+) \|", listed[-1], re.MULTILINE)
        return call("submit_pick", {"pick": first[1]})

    def sets(self):
        items: set[str] = set()
        chosen = []
        for mon in [mon for mon in self.owned if mon in PROBE_SETS][:6]:
            entry = dict(PROBE_SETS[mon]["set"])
            if entry["item"] in items:
                entry["item"] = ""
            items.add(entry["item"])
            chosen.append(entry)
        return chosen


@needs_harness
async def test_a_scripted_manager_finishes_a_season_on_the_real_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(league, "run_root", lambda: tmp_path / "runs")
    sample = Sample(
        id="probe",
        input=league.SAMPLE_INPUT,
        metadata={
            "seed": 7,
            "seats": [league.MODEL_SEAT, "random"],
            "focal_seat": league.MODEL_SEAT,
            "board": "regmc-202609",
            "transactions": True,
            "concurrency": 2,
        },
    )
    task = Task(
        dataset=MemoryDataset([sample]),
        solver=league.play_league(max_generations=6),
        scorer=[league_standing(), league_conduct()],
    )
    logs = await eval_async(task, model=offline_model(Probe()), log_dir=str(tmp_path / "logs"))
    result = logs[0].samples[0]
    assert result.error is None, result.error
    tools = {step["tool"] for step in result.store["steps"]}
    assert {"submit_pick", "submit_name", "submit_team", "submit_action", "submit_review"} <= tools
    assert {"submit_offer", "submit_free_agency"} <= tools
    standing = result.scores["league_standing"].value
    assert standing["completed"] == 1.0 and standing["playoffs"] == 1.0
    assert result.scores["league_conduct"].value["battle_decisions"] > 0
