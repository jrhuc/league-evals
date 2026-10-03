"""One VGC doubles game against a fixed opponent, played through the league harness's own state,
tools, and rendering. The model sees exactly what a league coach sees at each decision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolError, ToolParam, ToolParams
from inspect_ai.util import store

from .bridge import EXTERNAL, BridgeError, BridgeRejected, LeagueBridge
from .provenance import tree_hash
from .scorers import CALCULATORS, discipline, efficiency, mechanics, outcome

DATA = Path(__file__).resolve().parent / "data" / "pools"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "data" / "pools"

TOOL_ACCESS = ("full", "no_calculators")
BUDGET = (
    "Each decision allows up to {budget} replies, tool calls included; if none of them submits, "
    "the harness plays its default action for you."
)
NO_CALCULATORS = (
    "estimate_damage and compare_action_order are unavailable in this condition; instructions "
    "that mention them do not apply."
)
NUDGE = "Call submit_action with your choices for this decision."
BUDGET_WARNING = "{remaining} replies remain for this decision before the harness plays a default. Submit now unless a check is still essential."
CLOSED = "Decision already submitted. Wait for the next observation."
ACCEPTED = "Accepted. End your reply; the next observation follows."
DEFAULTED = "No decision was submitted in time; the harness played its default action."
SAMPLE_INPUT = "Play one game. Each observation shows the battle state and your legal menus."


@dataclass
class Decision:
    number: int
    turn: int
    phase: str
    error: str | None
    calls: list[str] = field(default_factory=list)
    generations: int = 0
    rejected: int = 0
    defaulted: bool = False
    rationale: str = ""
    post_submission_calls: int = 0
    tool_errors: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Session:
    bridge: LeagueBridge
    focal: str = "p1"
    access: str = "full"
    budget: int = 24
    decisions: list[Decision] = field(default_factory=list)
    exchange: int | None = None
    usage: dict[str, float] = field(default_factory=dict)
    rejection: str | None = None
    choice: str | None = None

    @property
    def current(self) -> Decision:
        return self.decisions[-1]

    def open_exchange(self) -> int:
        if self.exchange is None:
            self.current.post_submission_calls += 1
            raise ToolError(CLOSED)
        return self.exchange

    def resolved(self, row: dict[str, Any]) -> None:
        accepted = row["outcome"] == "accepted"
        self.choice = row["action"] if accepted else None
        self.rejection = None if accepted else row.get("showdown_error") or "rejected"


def load_pool(name: str) -> dict[str, Any]:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("pool must be a name, not a path")
    path = DATA / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m league_evals.pools {name}`")
    return json.loads(path.read_text())


def battle_samples(
    pool: dict[str, Any],
    seeds: list[int],
    opponent: str = "search",
    seats: tuple[str, ...] = ("p1", "p2"),
) -> list[Sample]:
    teams = pool["teams"]
    if len(teams) < 2 or len({t["id"] for t in teams}) != len(teams):
        raise ValueError("pool needs at least two teams with unique IDs")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    samples = []
    for seed in seeds:
        for i, first in enumerate(teams):
            for second in teams[i + 1 :]:
                for focal, other in ((first, second), (second, first)):
                    for seat in seats:
                        samples.append(_battle_sample(pool, focal, other, seed, opponent, seat))
    return samples


def _battle_sample(
    pool: dict[str, Any],
    focal: dict[str, str],
    other: dict[str, str],
    seed: int,
    opponent: str,
    seat: str,
) -> Sample:
    return Sample(
        id=f"{focal['id']}--{other['id']}--{seed}--{seat}",
        input=SAMPLE_INPUT,
        metadata={
            "format": pool["format"],
            "seed": seed,
            "focal_seat": seat,
            "focal_team": focal["id"],
            "focal_packed": focal["packed"],
            "opponent_team": other["id"],
            "opponent_packed": other["packed"],
            "opponent_policy": opponent,
            "pair": "--".join(sorted([focal["id"], other["id"]])),
            "pool_sha256": hashlib.sha256(
                json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "provenance": pool.get("provenance", {}),
        },
    )


def policy_seat(seats: list[str], seat: str) -> str:
    policies = [name for name in seats if name != EXTERNAL]
    if seat not in policies:
        raise BridgeError(f"opponent must be one of {', '.join(policies)}")
    return seat


PARAM_KEYS = set(ToolParam.model_fields) - {"additionalProperties"}
"""OpenAI's function schemas treat `additionalProperties: false` as strict mode, which then
demands every property be required; the harness's optional arguments are optional."""


def _param(schema: dict[str, Any]) -> ToolParam:
    kept: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in PARAM_KEYS:
            continue
        if key == "properties":
            kept[key] = {k: _param(v) for k, v in value.items() if isinstance(v, dict)}
        elif key == "items":
            kept[key] = _param(value)
        elif key == "anyOf":
            kept[key] = [_param(v) for v in value if isinstance(v, dict)]
        else:
            kept[key] = value
    return ToolParam(**kept)


def tool_params(schema: dict[str, Any]) -> ToolParams:
    properties = {
        k: _param(v) for k, v in schema.get("properties", {}).items() if isinstance(v, dict)
    }
    required = [name for name in schema.get("required", []) if name in properties]
    return ToolParams(type="object", properties=properties, required=required)


def harness_tool(definition: dict[str, Any], execute: Any) -> ToolDef:
    return ToolDef(
        execute,
        name=definition["name"],
        description=definition["description"],
        parameters=tool_params(definition["parameters"]),
        parallel=False,
    )


def reference_tool(session: Session, definition: dict[str, Any]) -> ToolDef:
    name = definition["name"]

    async def execute(**kwargs: Any) -> str:
        exchange = session.open_exchange()
        decision = session.current
        record = {"name": name, "arguments": kwargs}
        decision.trace.append(record)
        try:
            result = await session.bridge.request(
                "tool",
                {"pid": session.focal, "exchange": exchange, "name": name, "arguments": kwargs},
            )
        except BridgeRejected as error:
            record["error"] = str(error)
            raise ToolError(str(error)) from error
        record["result"] = result
        decision.calls.append(name)
        return result

    return harness_tool(definition, execute)


def submission_tool(session: Session, definition: dict[str, Any]) -> ToolDef:
    async def execute(**kwargs: Any) -> str:
        params = {"pid": session.focal, "exchange": session.open_exchange(), "input": kwargs}
        if session.usage:
            params["usage"] = session.usage
        try:
            await session.bridge.request("submit", params)
        except BridgeRejected as error:
            raise ToolError(str(error)) from error
        session.current.rationale = kwargs.get("rationale", "")
        session.exchange = None
        return ACCEPTED

    return harness_tool(definition, execute)


def system_prompt(league: str, budget: int, access: str) -> str:
    lines = [league, BUDGET.format(budget=budget)]
    if access == "no_calculators":
        lines.append(NO_CALCULATORS)
    return "\n".join(lines)


async def play_decision(
    state: TaskState, session: Session, event: dict[str, Any], note: str = ""
) -> None:
    model = get_model()
    exchange, view = event["exchange"], event["decision"]
    submission = exchange["submission"]
    catalog = [
        item
        for item in exchange["tools"]
        if session.access == "full" or item["name"] not in CALCULATORS
    ]
    tools = [reference_tool(session, item) for item in catalog]
    tools.append(submission_tool(session, submission))
    if not session.decisions:
        system = system_prompt(exchange["system"], session.budget, session.access)
        store().set("system_prompt", system)
        store().set("tool_catalog", catalog)
        state.messages = [
            ChatMessageSystem(content=system),
            ChatMessageUser(content=state.input_text),
        ]
    decision = Decision(
        number=int(exchange["task"].removeprefix("decision-")),
        turn=view["turn"],
        phase=view["phase"],
        error=session.rejection,
    )
    session.decisions.append(decision)
    session.exchange = exchange["id"]
    session.rejection = None
    session.usage = {}
    state.messages.append(ChatMessageUser(content=exchange["prompt"] + note))
    while session.exchange is not None and decision.generations < session.budget:
        decision.generations += 1
        output = await model.generate(state.messages, tools)
        state.output = output
        state.messages.append(output.message)
        usage = output.usage.model_dump(exclude_none=True) if output.usage else {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                session.usage[key] = session.usage.get(key, 0) + value
        if output.message.tool_calls:
            executed = await execute_tools(state.messages, tools)
            # Inspect can reject an invalid schema before entering the tool body.
            decision.rejected += sum(
                1
                for message in executed.messages
                if message.role == "tool"
                and message.function == submission["name"]
                and message.error is not None
            )
            decision.tool_errors += sum(
                1
                for message in executed.messages
                if message.role == "tool"
                and message.function != submission["name"]
                and message.error is not None
            )
            state.messages.extend(executed.messages)
        else:
            state.messages.append(ChatMessageUser(content=NUDGE))
        remaining = session.budget - decision.generations
        if session.exchange is not None and remaining in (3, 1):
            state.messages.append(
                ChatMessageUser(content=BUDGET_WARNING.format(remaining=remaining))
            )
    if session.exchange is not None:
        decision.defaulted = True
        await session.bridge.request(
            "abandon",
            {"pid": session.focal, "exchange": session.exchange, "reason": "reply budget spent"},
        )
        session.exchange = None
        state.messages.append(ChatMessageUser(content=DEFAULTED))


@solver
def play_battle(max_generations: int = 24, max_decisions: int = 200) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        bridge = await LeagueBridge.open(meta["format"])
        session = Session(
            bridge, focal=meta["focal_seat"], access=meta["tool_access"], budget=max_generations
        )
        other = "p2" if session.focal == "p1" else "p1"
        store().set("provenance", bridge.hello)
        store().set("task_sha256", tree_hash(Path(__file__).parent, ("*.py",)))
        store().set("completion", "incomplete")
        try:
            for key in ("harness_commit", "showdown_commit"):
                if meta["provenance"].get(key) != bridge.hello.get(key):
                    raise BridgeError(f"pool and engine {key} differ; re-export or restore engine")
            await bridge.request(
                "start",
                {
                    "seed": meta["seed"],
                    session.focal: {
                        "name": "focal",
                        "team": meta["focal_packed"],
                        "seat": EXTERNAL,
                    },
                    other: {
                        "name": "opponent",
                        "team": meta["opponent_packed"],
                        "seat": policy_seat(bridge.hello["seats"], meta["opponent_policy"]),
                    },
                    "policy_seed": meta["seed"],
                },
            )
            while (event := await bridge.next_event())["kind"] != "end":
                if event["kind"] == "decision":
                    session.resolved(event["row"])
                    continue
                if event["kind"] != "exchange":
                    raise BridgeError(f"unexpected bridge event: {event['kind']}")
                if len(session.decisions) >= max_decisions or state.completed:
                    store().set(
                        "completion", "decision_limit" if not state.completed else "interrupted"
                    )
                    return state
                await play_decision(state, session, event)
            result = event["outcome"]
            store().set("outcome", {k: v for k, v in result.items() if k != "log"})
            store().set("log", result["log"])
            if result["error"]:
                raise BridgeError("harness failed: " + result["error"])
            if any(line.startswith("|error|") for line in result["log"]):
                raise BridgeError("simulator failed: " + "\n".join(result["log"][-5:]))
            store().set("audit", await bridge.request("audit"))
            store().set("completion", "complete")
        except BaseException as error:
            store().set("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            store().set("decisions", [asdict(d) for d in session.decisions])
            await bridge.close()
        return state

    return solve


@task
def vgc_battle(
    pool: str = "test",
    seeds: str = "1",
    opponent: str = "search",
    max_generations: int = 24,
    tool_access: str = "full",
    focal_seat: str = "both",
    max_decisions: int = 200,
) -> Task:
    seed_list = [int(s) for s in str(seeds).split(",") if s.strip()]
    if max_generations < 1 or max_decisions < 1:
        raise ValueError("max_generations and max_decisions must be positive")
    if tool_access not in {*TOOL_ACCESS, "both"}:
        raise ValueError("tool_access must be full, no_calculators, or both")
    if focal_seat not in {"p1", "p2", "both"}:
        raise ValueError("focal_seat must be p1, p2, or both")
    seats = ("p1", "p2") if focal_seat == "both" else (focal_seat,)
    conditions = TOOL_ACCESS if tool_access == "both" else (tool_access,)
    samples = []
    for sample in battle_samples(load_pool(pool), seed_list, opponent, seats):
        for condition in conditions:
            samples.append(
                sample.model_copy(
                    update={
                        "id": f"{sample.id}--{condition}",
                        "metadata": {**sample.metadata, "tool_access": condition},
                    }
                )
            )
    return Task(
        dataset=MemoryDataset(samples, name=pool),
        solver=play_battle(max_generations=max_generations, max_decisions=max_decisions),
        scorer=[outcome(), mechanics(), discipline(), efficiency()],
        version=3,
        metadata={
            "pool": pool,
            "opponent_policy": opponent,
            "tool_access": tool_access,
            "focal_seat": focal_seat,
        },
    )
