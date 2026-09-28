"""One recorded VGC decision with a true or false attributed calculator claim."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from .notes import SOURCES, load_notes
from .position import POSITION_INPUT, play_position, position_samples
from .positions import load_positions
from .scorers import discipline, efficiency, note_trust, position_regret


@task
def vgc_note(
    notes: str = "league",
    sources: str = "own,agent,coach,harness",
    truths: str = "false,true",
    max_generations: int = 24,
    limit_hidden: int | None = None,
    min_turn: int | None = None,
) -> Task:
    source_list = [s.strip() for s in sources.split(",")]
    truth_list = [t.strip() for t in truths.split(",")]
    if any(s not in SOURCES for s in source_list) or len(set(source_list)) != len(source_list):
        raise ValueError(
            "sources must be a nonempty, unique selection of own, agent, coach, harness"
        )
    if any(t not in {"false", "true"} for t in truth_list) or len(set(truth_list)) != len(
        truth_list
    ):
        raise ValueError("truths must be a nonempty, unique selection of false, true")
    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    if limit_hidden is not None and limit_hidden < 0:
        raise ValueError("limit_hidden must be nonnegative")
    if min_turn is not None and min_turn < 1:
        raise ValueError("min_turn must be positive")
    claims = load_notes(notes)
    dataset = load_positions(claims["positions"])
    if claims["provenance"] != dataset["provenance"]:
        raise ValueError("notes and positions provenance differ; rebuild notes")
    base = {
        sample.id: sample
        for sample in position_samples(
            dataset, claims["positions"], limit_hidden=limit_hidden, min_turn=min_turn
        )
    }
    samples = []
    arms = [("control", None), *((t, s) for t in truth_list for s in source_list)]
    for claim in sorted(claims["claims"], key=lambda c: c["position"]):
        position_id = claim["position"]
        if position_id not in base:
            continue
        for truth, source in arms:
            samples.append(
                base[position_id].model_copy(
                    update={
                        "id": f"{position_id}--control"
                        if truth == "control"
                        else f"{position_id}--{truth}--{source}",
                        "input": POSITION_INPUT + " Notes may follow the observation.",
                        "metadata": {
                            **base[position_id].metadata,
                            "notes": notes,
                            "note": {
                                "truth": truth,
                                "source": source,
                                "text": None if truth == "control" else claim[f"{truth}_text"],
                                "claim": claim,
                            },
                        },
                    }
                )
            )
    return Task(
        dataset=MemoryDataset(samples, name=notes),
        solver=play_position(max_generations=max_generations),
        scorer=[position_regret(), note_trust(), discipline(), efficiency()],
        version=1,
        metadata={
            "notes": notes,
            "positions": claims["positions"],
            "sources": sources,
            "truths": truths,
            "tool_access": "full",
            "limit_hidden": limit_hidden,
            "min_turn": min_turn,
        },
    )
