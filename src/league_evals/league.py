"""A whole draft-league season: one model manages a franchise against fixed bot franchises through
the draft, a build before every series, every battle decision and post-game review, weekly
reviews, transaction windows, and the season review. Every task is the league harness's own
prompt, tools, and validation."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolError
from inspect_ai.util import store

from .battle import harness_tool, parse_seeds
from .bridge import BridgeError, BridgeRejected, LeagueBridge
from .provenance import tree_hash
from .scorers import league_conduct, league_standing

MODEL_SEAT = "external:model"
CONTROLS = ("bot", "random")
BATTLE_SUBMISSION = "submit_action"
SAMPLE_INPUT = "Manage one franchise through a whole draft-league season."
BUDGET = (
    "Each task allows up to {budget} replies, tool calls included. A battle decision that never "
    "submits plays the harness default; any other task that never submits ends the season."
)
NUDGE = "Call {tool} to complete this task."
ACCEPTED = "Accepted. End your reply; the next task follows."
CLOSED = "This task is already complete. Wait for the next one."
DEFAULTED = "No decision was submitted in time; the harness played its default action."
ELIDED = "\n[the rest of this earlier result is elided]"
KEPT_TOOL_CHARS = 400
"""Each task's prompt restates the state it needs, so a session keeps only the head of tool
results from its earlier tasks."""


def run_root() -> Path:
    configured = os.environ.get("LEAGUE_RUNS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "logs" / "league-runs"


@dataclass
class Step:
    session: str
    task: str
    tool: str
    generations: int = 0
    rejected: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    post_submission_calls: int = 0
    defaulted: bool = False
    seconds: float = 0.0


@dataclass
class Season:
    bridge: LeagueBridge
    budget: int
    sessions: dict[str, list[ChatMessage]] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


def elide(messages: list[ChatMessage]) -> None:
    for index, message in enumerate(messages):
        if isinstance(message, ChatMessageTool) and len(message.text) > KEPT_TOOL_CHARS:
            messages[index] = message.model_copy(
                update={"content": message.text[:KEPT_TOOL_CHARS] + ELIDED}
            )


async def play_task(season: Season, exchange: dict[str, Any]) -> None:
    started = time.monotonic()
    submission = exchange["submission"]
    step = Step(session=exchange["session"], task=exchange["task"], tool=submission["name"])
    season.steps.append(step)
    messages = season.sessions.get(step.session)
    if messages is None:
        system = exchange["system"] + "\n" + BUDGET.format(budget=season.budget)
        messages = season.sessions[step.session] = [ChatMessageSystem(content=system)]
    else:
        elide(messages)
    messages.append(ChatMessageUser(content=exchange["prompt"]))
    pending: dict[str, int | None] = {"exchange": exchange["id"]}
    usage: dict[str, float] = {}

    def open_exchange() -> int:
        if pending["exchange"] is None:
            step.post_submission_calls += 1
            raise ToolError(CLOSED)
        return pending["exchange"]

    def reference(definition: dict[str, Any]) -> ToolDef:
        name = definition["name"]

        async def execute(**kwargs: Any) -> str:
            params = {"exchange": open_exchange(), "name": name, "arguments": kwargs}
            step.tool_calls += 1
            try:
                return await season.bridge.request("tool", params)
            except BridgeRejected as error:
                raise ToolError(str(error)) from error

        return harness_tool(definition, execute)

    async def submit(**kwargs: Any) -> str:
        params = {"exchange": open_exchange(), "input": kwargs, "usage": usage}
        try:
            await season.bridge.request("submit", params)
        except BridgeRejected as error:
            raise ToolError(str(error)) from error
        pending["exchange"] = None
        return ACCEPTED

    tools = [reference(item) for item in exchange["tools"]]
    tools.append(harness_tool(submission, submit))
    model = get_model()
    while pending["exchange"] is not None and step.generations < season.budget:
        step.generations += 1
        output = await model.generate(messages, tools)
        messages.append(output.message)
        for key, value in (
            output.usage.model_dump(exclude_none=True) if output.usage else {}
        ).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        if not output.message.tool_calls:
            messages.append(ChatMessageUser(content=NUDGE.format(tool=submission["name"])))
            continue
        executed = await execute_tools(messages, tools)
        for message in executed.messages:
            if message.role != "tool" or message.error is None:
                continue
            if message.function == submission["name"]:
                step.rejected += 1
            else:
                step.tool_errors += 1
        messages.extend(executed.messages)
    step.seconds = time.monotonic() - started
    if pending["exchange"] is None:
        return
    if submission["name"] != BATTLE_SUBMISSION:
        raise BridgeError(f"{step.task} got no accepted submission in {season.budget} replies")
    step.defaulted = True
    await season.bridge.request(
        "abandon", {"exchange": pending["exchange"], "reason": "reply budget spent"}
    )
    messages.append(ChatMessageUser(content=DEFAULTED))


async def run_season(season: Season, focal: str) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    working: set[asyncio.Task[None]] = set()
    events = asyncio.ensure_future(season.bridge.next_event())
    try:
        while True:
            done, _ = await asyncio.wait({events, *working}, return_when=asyncio.FIRST_COMPLETED)
            for finished in done - {events}:
                working.discard(finished)
                finished.result()
            if events not in done:
                continue
            event = events.result()
            if event["kind"] == "end":
                return event["outcome"]
            if event["kind"] == "series":
                series.append(event["series"])
                store().set("series", series)
            elif event["kind"] == "exchange" and event["exchange"]["model"] == focal:
                working.add(asyncio.create_task(play_task(season, event["exchange"])))
            else:
                raise BridgeError(f"unexpected bridge event: {event['kind']}")
            events = asyncio.ensure_future(season.bridge.next_event())
    finally:
        for pending in (events, *working):
            pending.cancel()


@solver
def play_league(max_generations: int = 40) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        run_dir = run_root() / f"{state.sample_id}-{time.strftime('%Y%m%dT%H%M%S')}"
        store().set("run_dir", str(run_dir))
        store().set("task_sha256", tree_hash(Path(__file__).parent, ("*.py",)))
        store().set("completion", "incomplete")
        bridge = await LeagueBridge.league(
            {
                "seats": meta["seats"],
                "seed": meta["seed"],
                "run_dir": str(run_dir),
                "board": meta["board"],
                "transactions": meta["transactions"],
                "concurrency": meta["concurrency"],
            }
        )
        season = Season(bridge, budget=max_generations)
        store().set("provenance", bridge.hello)
        try:
            outcome = await run_season(season, meta["focal_seat"])
            store().set("outcome", outcome)
            store().set("focal_entrant", outcome["entrants"].index(meta["focal_seat"]))
            if outcome["error"]:
                raise BridgeError("harness failed: " + outcome["error"])
            store().set("completion", "complete")
        except BaseException as error:
            store().set("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            store().set("steps", [asdict(step) for step in season.steps])
            await bridge.close()
        return state

    return solve


def league_samples(
    seeds: list[int], focal: str, bots: int, board: str, transactions: bool, concurrency: int
) -> list[Sample]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if focal not in (MODEL_SEAT, *CONTROLS):
        raise ValueError(f"focal must be {MODEL_SEAT} or one of {', '.join(CONTROLS)}")
    if bots < 1:
        raise ValueError("a league needs at least one bot")
    return [
        Sample(
            id=f"{board}--{bots + 1}-seats--{seed}--{focal.removeprefix('external:')}",
            input=SAMPLE_INPUT,
            metadata={
                "seed": seed,
                "seats": [focal, *["bot"] * bots],
                "focal_seat": focal,
                "board": board,
                "transactions": transactions,
                "concurrency": concurrency,
            },
        )
        for seed in seeds
    ]


@task
def vgc_league(
    seeds: str | list[str] = "1",
    bots: int = 3,
    board: str = "regmc-202609",
    transactions: bool = True,
    concurrency: int = 4,
    control: str = "",
    max_generations: int = 40,
) -> Task:
    """`control=bot` or `control=random` puts that fixed policy in the focal seat, so the same
    pipeline measures what a franchise without a model achieves."""
    seed_list = parse_seeds(seeds)
    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    focal = control or MODEL_SEAT
    return Task(
        dataset=MemoryDataset(
            league_samples(seed_list, focal, bots, board, transactions, concurrency), name=board
        ),
        solver=play_league(max_generations=max_generations),
        scorer=[league_standing(), league_conduct()],
        version=1,
        metadata={"board": board, "bots": bots, "transactions": transactions, "control": control},
    )
