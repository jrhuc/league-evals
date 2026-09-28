"""One recorded VGC decision, scored against independent offline rollouts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from .battle import (
    INSTRUCTIONS,
    TOOL_INSTRUCTIONS,
    Session,
    play_decision,
    reference_tool,
    submit_action,
)
from .bridge import BridgeError, LeagueBridge
from .positions import load_positions
from .provenance import tree_hash
from .scorers import CALCULATORS, discipline, efficiency, position_regret

POSITION_INPUT = (
    "You are taking over one game that is already under way. The observation shows the battle "
    "state, your legal menus, and everything that has happened so far. Make this one decision "
    "so as to maximize your probability of winning the game."
)


async def resume_position(bridge: LeagueBridge, source: dict, position: dict) -> dict:
    focal = position["focal"]
    other = "p2" if focal == "p1" else "p1"
    event = await bridge.request(
        "start",
        {
            "format": source["format"],
            "seed": source["seed"],
            focal: {"name": "focal", "team": source["packed"][focal]},
            other: {"name": "opponent", "team": source["packed"][other]},
            "external": [focal],
            "opponent": "greedy",
            "script": {
                focal: source["choices"][focal][: position["choice_index"]],
                other: source["choices"][other][: position["opponent_choice_index"]],
            },
        },
    )
    if (
        event.get("kind") != "decision"
        or event.get("phase") != "turn"
        or event.get("turn") != position["turn"]
        or event.get("decision") != 1
        or event.get("pid", focal) != focal
    ):
        raise BridgeError(f"unexpected first position event: {event}")
    return event


@solver
def play_position(max_generations: int = 24) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        position = meta["position"]
        source = meta["dataset"]["games"][position["game"]]["source"]
        access = meta["tool_access"]
        store().set("completion", "incomplete")
        store().set("position_choice", None)
        session = None
        bridge = None
        try:
            bridge = await LeagueBridge.open(source["format"], "open")
            session = Session(bridge, focal=position["focal"])
            store().set("provenance", bridge.hello)
            store().set("task_sha256", tree_hash(Path(__file__).parent, ("*.py",)))
            for key in ("harness_commit", "showdown_commit"):
                if meta["dataset"]["provenance"].get(key) != bridge.hello.get(key):
                    raise BridgeError(
                        f"positions and engine {key} differ; re-export or restore engine"
                    )
            catalog = await bridge.request("tools")
            if access == "no_calculators":
                catalog = [item for item in catalog if item["name"] not in CALCULATORS]
            tools = [reference_tool(session, item) for item in catalog] + [submit_action(session)]
            system = INSTRUCTIONS.format(
                budget=max_generations, tool_instructions=TOOL_INSTRUCTIONS[access]
            )
            store().set("system_prompt", system)
            store().set("tool_catalog", catalog)
            state.messages = [
                ChatMessageSystem(content=system),
                ChatMessageUser(content=state.input_text),
            ]
            event = await resume_position(bridge, source, position)
            note = meta.get("note", {})
            if note.get("text"):
                from .notes import note_block

                event = {
                    **event,
                    "prompt": event["prompt"] + note_block(note["source"], note["text"]),
                }
            await play_decision(state, session, event, tools, max_generations)
            store().set("position_choice", session.choice)
            store().set("completion", "complete")
        except BaseException as error:
            store().set("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            store().set("decisions", [asdict(d) for d in session.decisions] if session else [])
            if bridge is not None:
                await bridge.close()
        return state

    return solve


def position_samples(
    dataset: dict,
    positions: str,
    conditions: tuple[str, ...] = ("full",),
    limit_hidden: int | None = None,
    min_turn: int | None = None,
) -> list[Sample]:
    samples = []
    for position in dataset["positions"]:
        if limit_hidden is not None and position["hidden_opponents"] > limit_hidden:
            continue
        if min_turn is not None and position["turn"] < min_turn:
            continue
        position = {**position, "good_margin": dataset["selection"]["good_margin"]}
        for condition in conditions:
            samples.append(
                Sample(
                    id=f"{position['id']}--{condition}" if len(conditions) > 1 else position["id"],
                    input=POSITION_INPUT,
                    metadata={
                        "positions": positions,
                        "position": position,
                        "dataset": {
                            **{k: dataset[k] for k in ("id", "selection", "provenance")},
                            "games": {position["game"]: dataset["games"][position["game"]]},
                        },
                        "recorded_model": position["recorded"]["model"],
                        "game": position["game"],
                        "pair": position["game"],
                        "tool_access": condition,
                        "format": dataset["format"],
                        "focal_seat": position["focal"],
                    },
                )
            )
    return samples


@task
def vgc_position(
    positions: str = "league",
    tool_access: str = "full",
    max_generations: int = 24,
    limit_hidden: int | None = None,
    min_turn: int | None = None,
) -> Task:
    if tool_access not in {*TOOL_INSTRUCTIONS, "both"}:
        raise ValueError("tool_access must be full, no_calculators, or both")
    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    if limit_hidden is not None and limit_hidden < 0:
        raise ValueError("limit_hidden must be nonnegative")
    if min_turn is not None and min_turn < 1:
        raise ValueError("min_turn must be positive")
    dataset = load_positions(positions)
    conditions = tuple(TOOL_INSTRUCTIONS) if tool_access == "both" else (tool_access,)
    samples = position_samples(dataset, positions, conditions, limit_hidden, min_turn)
    return Task(
        dataset=MemoryDataset(samples, name=positions),
        solver=play_position(max_generations=max_generations),
        scorer=[position_regret(), discipline(), efficiency()],
        version=1,
        metadata={
            "positions": positions,
            "tool_access": tool_access,
            "limit_hidden": limit_hidden,
            "min_turn": min_turn,
        },
    )
