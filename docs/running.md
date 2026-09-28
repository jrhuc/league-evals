# Running and reading evaluations

## Engine and installation

Run the [README setup](../README.md) from a checkout. It builds the upstream commit and
checksum-verified patch in [engine.lock.json](../engine.lock.json), checks the patched source,
and writes a local build manifest. No global `vp` CLI is needed. An isolated network install
and real-engine integration tests have been exercised locally; CI repeats the build on Linux.

`LEAGUE_DIR` selects another built `packages/league` checkout. Samples record source, compiled
JavaScript, and compiled Showdown fingerprints. `build_verified` indicates that those bytes match
the local build manifest. An alternate build without a manifest is labelled unverified. This is
a consistency check, not independent attestation. Pool/engine commit mismatches fail before model
generation. `uv run python -m league_evals.pools test` intentionally regenerates the team artifact.

Python tests run without Node; real-engine tests skip when the bridge is absent. CI has a separate
engine build/integration job. Packaged wheels include the team pool; an installed wheel needs
`LEAGUE_DIR` pointing to a built engine.

## Task options

| Option | Default | Meaning |
| --- | --- | --- |
| `pool` | `test` | Packed team artifact |
| `seeds` | `1` | Unique comma-separated simulator/opponent seeds |
| `opponent` | `greedy` | Fixed damage policy or seeded `random` |
| `focal_seat` | `both` | `p1`, `p2`, or balanced seats |
| `tool_access` | `full` | `full`, `no_calculators`, or `both` for a matched experiment |
| `max_generations` | `24` | Replies per decision; warnings at 3 and 1 remaining |
| `max_decisions` | `200` | Cutoff; unfinished games stay incomplete |
| `sheets` | `open` | Only supported setting; the format reveals sheets |
| `league_prompt` | `false` | Parent-league prompt; requires `full` and is a different condition |

Each seed creates 60 games for one tool condition or 120 for both. `--epochs` repeats the full
matrix with fresh provider generations; it does not supply a provider sampling seed. `--limit`
is a plumbing check and can leave unmatched cells. Some OpenAI-compatible providers need
`-M strict_tools=false` because the inherited schemas include optional parameters. OpenRouter
reserves each in-flight request's worst-case cost against the account balance; with a small balance,
lower `--max-connections` or runs stop on `402 in_flight_budget_exhausted`.

A reply-budget timeout plays a logged default. Decision/sample limits produce incomplete records.
Model-correctable tool errors can be retried within the decision. Bridge timeouts, malformed
responses, and simulator failures produce sample errors with partial traces.

## Position task

`vgc_position` scores one recorded decision per sample against offline rollouts; see
[Position evaluation](positions.md) for the method and limits.

| Option | Default | Meaning |
| --- | --- | --- |
| `positions` | `league` | Dataset under `data/positions/` |
| `tool_access` | `full` | `full`, `no_calculators`, or `both`, as in `vgc_battle` |
| `max_generations` | `24` | Replies for the decision; a timeout plays a logged default |
| `limit_hidden` | unset | Keep decisions with at most this many unseen opposing Pokémon |
| `min_turn` | unset | Keep decisions from this turn on; early values lean hardest on the continuation |

`league-positions build|baselines|report` builds a dataset from a league run, prints
the no-model baselines, and summarises `vgc_position` logs. A choice outside a decision's shortlist
is valued at scoring time, which needs the built engine.

## Note task

`vgc_note` attaches a true or false calculator claim with a stated source to a recorded decision; see
[Note trust](notes.md).

| Option | Default | Meaning |
| --- | --- | --- |
| `notes` | `league` | Claims under `data/notes/`; they name the positions dataset they were built on (`league-all`) |
| `sources` | `own,agent,coach,harness` | Stated sources to include |
| `truths` | `false,true` | Claim arms to include; the control arm is always present |
| `max_generations`, `limit_hidden`, `min_turn` | as `vgc_position` | |

Samples are ordered by position with the control first: with the defaults, `--limit 9` is one complete
position. `league-notes build|report` builds the claims and summarises logs by arm.

## Reports

```sh
uv run python -m league_evals.report logs/pilot
uv run python -m league_evals.report logs/pilot --json > results.json
uv run python -m league_evals.report logs/pilot --compare NO_CALCULATORS_ID,FULL_ID
# Prices are USD per million tokens: uncached input, output, cache read, cache write.
uv run python -m league_evals.report logs/pilot --price 'PROVIDER/MODEL=IN,OUT,READ,WRITE'
uv run inspect view --log-dir logs/pilot
```

The table separates actual tool conditions, task versions, prompts, generation settings, budgets,
and engine fingerprints. Its win means weight each observed unordered pair equally and condition
on completion. Unassisted wins exclude any harness defaults or simulator substitutions.
Inspect's own aggregate display pools the dataset; use this report to compare the two arms.

The comparison shows full-minus-withheld outcomes on matched cells, disagreement counts,
per-pair effects, and missing-outcome bounds. It rejects configuration confounds, duplicate cells
including failed-then-successful attempts, and samples with recorded retries. It cannot reconstruct
unstarted cells or discarded retry costs. The JSON export includes record paths and diagnostics.

Cost uses recorded provider totals when available, otherwise explicit rates for every used token
category. Unknown prices stay unknown. Reasoning tokens already included in output tokens are not
charged a second time. Retained failed samples contribute cost; discarded retries can cost more.

## Reproduce controls and inspect legacy evidence

```sh
uv run python -m league_evals.baseline --seeds 1,2,3 --opponent greedy --output default-greedy.jsonl
uv run python -m league_evals.baseline --seeds 1,2,3 --opponent random --output default-random.jsonl
uv run python -m league_evals.baseline --summarise default-greedy.jsonl default-random.jsonl
uv run inspect view --log-dir examples/logs
```

The baseline command refuses to overwrite an existing output. Raw simulator hashes identify the
recorded bytes, including wall-clock `|t:|` lines. Seed replay compares actions and simulator lines
with those timestamps removed; the model's choices remain stochastic.

The eight legacy model games used a different prompt/version and unequal matchup coverage. One
win used a fallback. Their provenance has Git HEADs but lacks runtime fingerprints. Keep them
separate from the pilot and retain their original logs as historical smoke evidence.
