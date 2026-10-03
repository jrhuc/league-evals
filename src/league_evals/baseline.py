"""No-model negative control: the harness default on the same cells as vgc_battle."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .battle import battle_samples, load_pool, policy_seat
from .bridge import EXTERNAL, FORMAT, BridgeError, LeagueBridge, request_sync
from .scorers import outcome_summary


def summarise(paths: list[Path]) -> str:
    lines = [
        "| opponent | attempts | complete | pairs | wins | win rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        done = [r for r in rows if r["complete"]]
        policies = {r["metadata"]["opponent_policy"] for r in rows}
        if len(policies) != 1:
            raise ValueError(f"{path} mixes opponent policies")
        wins = sum(r["scores"]["win"] for r in done)
        pairs = {r["metadata"]["pair"] for r in done}
        rate = f"{wins / len(done):.1%}" if done else "undefined"
        lines.append(
            f"| {next(iter(policies))} | {len(rows)} | {len(done)} | {len(pairs)} | {wins:.0f} | {rate} |"
        )
    return "\n".join(lines)


async def default_game(metadata: dict, max_decisions: int = 200) -> dict:
    bridge = await LeagueBridge.open(metadata["format"])
    focal = metadata["focal_seat"]
    opponent = "p2" if focal == "p1" else "p1"
    try:
        await bridge.request(
            "start",
            {
                "seed": metadata["seed"],
                focal: {"name": "focal", "team": metadata["focal_packed"], "seat": EXTERNAL},
                opponent: {
                    "name": "opponent",
                    "team": metadata["opponent_packed"],
                    "seat": policy_seat(bridge.hello["seats"], metadata["opponent_policy"]),
                },
                "policy_seed": metadata["seed"],
            },
        )
        decisions = 0
        actions = []
        result = None
        while result is None:
            event = await bridge.next_event()
            if event["kind"] == "end":
                result = event["outcome"]
            elif event["kind"] == "decision":
                if event["row"]["outcome"] == "accepted":
                    actions.append(event["row"]["action"])
            elif decisions == max_decisions:
                break
            else:
                decisions += 1
                await bridge.request(
                    "abandon",
                    {"pid": focal, "exchange": event["exchange"]["id"], "reason": "default policy"},
                )
        if result and result["error"]:
            raise BridgeError("harness failed: " + result["error"])
        if result and any(line.startswith("|error|") for line in result["log"]):
            raise BridgeError("simulator failed: " + str(result["log"]))
        return {
            "policy": "harness-default",
            "metadata": metadata,
            "complete": result is not None,
            "decisions": decisions,
            "actions": actions,
            "scores": outcome_summary(result, focal) if result else None,
            "outcome": result,
            "provenance": bridge.hello,
        }
    finally:
        await bridge.close()


async def run(args) -> None:
    samples = battle_samples(
        load_pool(args.pool), [int(s) for s in args.seeds.split(",")], args.opponent
    )
    slots = asyncio.Semaphore(args.concurrency)

    async def one(sample):
        async with slots:
            try:
                return await default_game(sample.metadata)
            except (BridgeError, OSError, ValueError, TimeoutError) as error:
                return {
                    "policy": "harness-default",
                    "metadata": sample.metadata,
                    "complete": False,
                    "error": f"{type(error).__name__}: {error}",
                }

    # Exclusive creation makes an accidental rerun unable to overwrite evidence.
    with args.output.open("x") as output:
        completed = 0
        wins = 0
        tasks = [asyncio.create_task(one(sample)) for sample in samples]
        try:
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                row = await task
                output.write(json.dumps(row) + "\n")
                output.flush()
                completed += row["complete"]
                wins += (row.get("scores") or {}).get("win", 0)
                if i % 20 == 0 or i == len(samples):
                    print(
                        f"{i}/{len(samples)} attempts; {completed} complete; {int(wins)} default-policy wins",
                        flush=True,
                    )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="test")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--opponent", default="search")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarise", nargs="+", type=Path, metavar="JSONL")
    args = parser.parse_args()
    if args.summarise:
        print(summarise(args.summarise))
        return
    if args.output is None:
        parser.error("--output is required when running games")
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    try:
        (hello,) = request_sync([("open", {"format": FORMAT})])
        policy_seat(hello["seats"], args.opponent)
    except BridgeError as error:
        parser.error(str(error))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
