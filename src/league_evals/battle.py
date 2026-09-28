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
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolParam, ToolParams, tool
from inspect_ai.util import store

from .bridge import BridgeError, BridgeRejected, LeagueBridge
from .provenance import tree_hash
from .scorers import CALCULATORS, discipline, efficiency, mechanics, outcome

DATA = Path(__file__).resolve().parent / "data" / "pools"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "data" / "pools"

INSTRUCTIONS = """\
You are an expert VGC (doubles) player in a single game. Maximize your probability of winning it.
The pinned Pokémon Showdown simulator decides legality and outcomes; this format may postdate \
your training data, so trust the rules and menus you are given over remembered mechanics.
Choose only from the numbered menus. Never invent a move, target, switch, effect, immunity, \
stat, or revealed fact.
{tool_instructions}
Both active Pokémon are one joint decision. Within a turn, switches resolve first, then Mega \
Evolutions in Speed order, then moves by priority and then Speed. One Mega Evolution is allowed \
per game.
When ready, call submit_action with one menu index per displayed slot (at team preview: the \
ordered picks, leads first) and a short rationale. After it is accepted, end your reply and wait \
for the next observation. Put submit_action last; later calls in the same reply are rejected.
Each decision allows up to {budget} replies, tool calls included; \
several tool calls fit in one reply. If none of them submits, the harness plays a default for you.
"""
TOOL_INSTRUCTIONS = {
    "full": "estimate_damage and compare_action_order are available to check damage and action "
    "order. They use this format's engine with the visible state and stated assumptions. "
    "Trust a result only for the factors it says it applied. Use checks when useful to your "
    "decision; several calls can be made in one reply. lookup_* and calculate_stats read rules.",
    "no_calculators": "Damage and action-order calculators are unavailable in this condition. "
    "lookup_* and calculate_stats are available for rules and stat reference. Use the visible "
    "state, reference tools, and your reasoning to choose actions.",
}
NUDGE = "Call submit_action with your choices for this decision."
BUDGET_WARNING = "{remaining} replies remain for this decision before the harness plays a default. Submit now unless a check is still essential."
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
    decisions: list[Decision] = field(default_factory=list)
    next_event: dict[str, Any] | None = None
    choice: str | None = None

    @property
    def current(self) -> Decision:
        return self.decisions[-1]


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
    opponent: str = "greedy",
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


def reference_tool(session: Session, definition: dict[str, Any]) -> ToolDef:
    name = definition["name"]

    async def execute(**kwargs: Any) -> str:
        decision = session.current
        if session.next_event is not None:
            decision.post_submission_calls += 1
            raise ToolError("Decision already submitted. Wait for the next observation.")
        record = {"name": name, "arguments": kwargs}
        decision.trace.append(record)
        try:
            result = await session.bridge.request(
                "call", {"pid": session.focal, "name": name, "arguments": kwargs}
            )
        except BridgeRejected as error:
            record["error"] = str(error)
            raise ToolError(str(error)) from error
        record["result"] = result
        decision.calls.append(name)
        return result

    return ToolDef(
        execute,
        name=name,
        description=definition["description"],
        parameters=tool_params(definition["parameters"]),
        parallel=False,
    )


@tool
def submit_action(session: Session) -> Tool:
    async def execute(choices: list[int], rationale: str = "") -> str:
        """Submit the joint decision for the current observation.

        Args:
            choices: One zero-based menu index per displayed slot, in slot order. At team
                preview, the four ordered picks: leads first, then the back.
            rationale: Your final reason for this decision, kept in your private record.
        """
        if session.next_event is not None:
            session.current.post_submission_calls += 1
            raise ToolError("Decision already submitted. Wait for the next observation.")
        try:
            reply = await session.bridge.request(
                "submit", {"pid": session.focal, "choices": choices}
            )
        except BridgeRejected as error:
            raise ToolError(str(error)) from error
        session.current.rationale = rationale
        session.next_event = reply["next"]
        session.choice = reply["choice"]
        return f"Accepted: {reply['choice']}. End your reply; the next observation follows."

    return execute


def turn_message(event: dict[str, Any]) -> str:
    text = event["prompt"]
    if event["error"]:
        text += f"\nThe simulator rejected the previous action: {event['error']}"
    return text


async def play_decision(
    state: TaskState,
    session: Session,
    event: dict[str, Any],
    tools: list[Tool | ToolDef],
    max_generations: int,
) -> None:
    model = get_model()
    decision = Decision(
        number=event["decision"],
        turn=event["turn"],
        phase=event["phase"],
        error=event["error"],
    )
    session.decisions.append(decision)
    session.next_event = None
    session.choice = None
    state.messages.append(ChatMessageUser(content=turn_message(event)))
    while session.next_event is None and decision.generations < max_generations:
        decision.generations += 1
        output = await model.generate(state.messages, tools)
        state.output = output
        state.messages.append(output.message)
        if output.message.tool_calls:
            executed = await execute_tools(state.messages, tools)
            # Inspect can reject an invalid schema before entering the tool body.
            decision.rejected += sum(
                1
                for message in executed.messages
                if message.role == "tool"
                and message.function == "submit_action"
                and message.error is not None
            )
            decision.tool_errors += sum(
                1
                for message in executed.messages
                if message.role == "tool"
                and message.function != "submit_action"
                and message.error is not None
            )
            state.messages.extend(executed.messages)
        else:
            state.messages.append(ChatMessageUser(content=NUDGE))
        remaining = max_generations - decision.generations
        if session.next_event is None and remaining in (3, 1):
            state.messages.append(
                ChatMessageUser(content=BUDGET_WARNING.format(remaining=remaining))
            )
    if session.next_event is None:
        decision.defaulted = True
        reply = await session.bridge.request("submit", {"pid": session.focal, "choices": "default"})
        session.next_event = reply["next"]
        session.choice = reply["choice"]
        state.messages.append(
            ChatMessageUser(
                content=f"No decision was submitted in time; the harness played the default: {reply['choice']}."
            )
        )


@solver
def play_battle(
    sheets: str = "open",
    max_generations: int = 24,
    league_prompt: bool = False,
    tool_access: str = "full",
    max_decisions: int = 200,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        access = meta.get("tool_access", tool_access)
        bridge = await LeagueBridge.open(meta["format"], sheets)
        session = Session(bridge, focal=meta["focal_seat"])
        store().set("provenance", bridge.hello)
        store().set("task_sha256", tree_hash(Path(__file__).parent, ("*.py",)))
        store().set("completion", "incomplete")
        try:
            for key in ("harness_commit", "showdown_commit"):
                if meta["provenance"].get(key) != bridge.hello.get(key):
                    raise BridgeError(f"pool and engine {key} differ; re-export or restore engine")
            catalog = await bridge.request("tools")
            if access == "no_calculators":
                catalog = [item for item in catalog if item["name"] not in CALCULATORS]
            tools = [reference_tool(session, item) for item in catalog] + [submit_action(session)]
            system = (
                await bridge.request("system")
                if league_prompt
                else INSTRUCTIONS.format(
                    budget=max_generations, tool_instructions=TOOL_INSTRUCTIONS[access]
                )
            )
            if league_prompt:
                system += f"\nEach decision allows {max_generations} replies. Submit last."
            store().set("system_prompt", system)
            store().set("tool_catalog", catalog)
            state.messages = [
                ChatMessageSystem(content=system),
                ChatMessageUser(content=state.input_text),
            ]
            event = await bridge.request(
                "start",
                {
                    "format": meta["format"],
                    "seed": meta["seed"],
                    session.focal: {"name": "focal", "team": meta["focal_packed"]},
                    "p2" if session.focal == "p1" else "p1": {
                        "name": "opponent",
                        "team": meta["opponent_packed"],
                    },
                    "external": [session.focal],
                    "opponent": meta["opponent_policy"],
                    "opponent_seed": meta["seed"],
                },
            )
            while event["kind"] == "decision":
                if len(session.decisions) >= max_decisions or state.completed:
                    store().set(
                        "completion", "decision_limit" if not state.completed else "interrupted"
                    )
                    return state
                await play_decision(state, session, event, tools, max_generations)
                event = session.next_event
            if event["kind"] != "end":
                raise BridgeError(f"unexpected bridge event: {event['kind']}")
            result = event["outcome"]
            store().set("outcome", {k: v for k, v in result.items() if k != "log"})
            store().set("log", result["log"])
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
    opponent: str = "greedy",
    sheets: str = "open",
    max_generations: int = 24,
    league_prompt: bool = False,
    tool_access: str = "full",
    focal_seat: str = "both",
    max_decisions: int = 200,
) -> Task:
    seed_list = [int(s) for s in str(seeds).split(",") if s.strip()]
    if opponent not in {"random", "greedy"}:
        raise ValueError("opponent must be random or greedy")
    if sheets != "open":
        raise ValueError("only open sheets are supported: the pinned format reveals team sheets")
    if max_generations < 1 or max_decisions < 1:
        raise ValueError("max_generations and max_decisions must be positive")
    if tool_access not in {*TOOL_INSTRUCTIONS, "both"}:
        raise ValueError("tool_access must be full, no_calculators, or both")
    if league_prompt and tool_access != "full":
        raise ValueError("league_prompt requires full tools; its instructions assume calculators")
    if focal_seat not in {"p1", "p2", "both"}:
        raise ValueError("focal_seat must be p1, p2, or both")
    seats = ("p1", "p2") if focal_seat == "both" else (focal_seat,)
    conditions = tuple(TOOL_INSTRUCTIONS) if tool_access == "both" else (tool_access,)
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
        solver=play_battle(
            sheets=sheets,
            max_generations=max_generations,
            league_prompt=league_prompt,
            tool_access=tool_access,
            max_decisions=max_decisions,
        ),
        scorer=[outcome(), mechanics(), discipline(), efficiency()],
        version=2,
        metadata={
            "pool": pool,
            "sheets": sheets,
            "opponent_policy": opponent,
            "tool_access": tool_access,
            "focal_seat": focal_seat,
        },
    )
