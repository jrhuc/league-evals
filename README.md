# league-evals

[Inspect](https://inspect.aisi.org.uk) evaluations of a model playing Pokémon VGC doubles through
the [AI Draft League](https://github.com/jrhuc/ai-draft-league) harness, from one decision up to a
whole season. Each asks one question, is scored by the pinned simulator with no judge model, and
keeps every decision, tool call, and log it was scored from.

VGC supplies joint actions, hidden information, and stochastic outcomes inside a reproducible
simulator, and the league supplies real games between frontier models: 27 of one season's 28 games
replay line for line from their seed and recorded choices, so a fresh model can be dropped into any
turn of them.

| Task | Question | One sample is |
| --- | --- | --- |
| [`vgc_position`](docs/positions.md) | Where the highest-damage action is not the highest-win-rate action, which does a model choose? | one recorded decision, scored against offline rollouts |
| [`vgc_note`](docs/notes.md) | When a notebook line is wrong, does the model re-run the calculation or act on it, and does that depend on who the note says wrote it? | the same kind of decision, with a notebook excerpt attached |
| [`vgc_battle`](docs/design.md) | Does calculator access improve matched game outcomes? | one whole game against a fixed opponent |
| [`vgc_league`](docs/running.md#league-task) | Where does a model finish when it manages a franchise for a whole season against bot franchises? | one season: draft, builds, every game, reviews, and trades |

The first three evaluate a model plus its tools in one game or one decision. `vgc_league` is the
long-horizon one: the model drafts against bots, builds a six for every opponent, plays every game,
reviews between weeks, and may trade, and it is scored by where it finishes. Transfer to other domains
is untested here.

## What we know so far

One fresh model has been run, on the position task. Everything else below cost no model spend.

**Positions.** 157 of 494 valued decisions are kept: those where the reference action beats the
damage-greedy one on independent dice. Scored on a third, independent set of rollouts:

| No-model reference | Decisions | Mean win rate given up | Within 0.10 of the reference | Beats the damage policy |
| --- | --- | --- | --- | --- |
| Damage-greedy action | 157 | 0.47 | 3.2% | 0.0% |
| Uniformly random legal action | 157 | 0.50 | – | – |
| What the league model actually played | 157 | 0.23 | 39.5% | 77.1% |
| The engine's rollout search opponent | 157 | 0.18 | 48.4% | 82.8% |

The last row is the `search` policy `vgc_battle` now plays against, asked for its own action on each
decision. It reads the simulator's battle, so it knows the opposing bench, and it shares the damage
continuation the scoring rollouts assume; both favour it.

The rollouts assume a continuation nobody played, so they were checked against what happened: the
value of the action a league model really took predicts who won that game with AUC 0.89 over all 494
decisions, 0.96 from turn 6, and 0.68 on turns 1-2, where the assumption bites hardest.
[Method, validity check, and limits](docs/positions.md).

**First model run.** Claude Opus 5.5 with full calculator access, through OpenRouter, on the 99 kept
decisions from turn 3 on. This used version 1 of the task, with this repository's own prompt; the
task now sends the league coach's prompt, so new runs are a different condition. It cost about $13.60, including requests lost to two interrupted
attempts. On those 99 decisions:

| Policy | Mean win rate given up | Within 0.10 of the reference | Beats the damage policy |
| --- | --- | --- | --- |
| Damage-greedy action | 0.45 | 2.0% | 0.0% |
| Uniformly random legal action | 0.49 | – | – |
| What the league model actually played | 0.22 | 42.4% | 75.8% |
| The engine's rollout search opponent | 0.18 | 53.5% | 80.8% |
| Claude Opus 5.5 | 0.23 | 39.4% | 74.7% |

Paired with the league's move on each decision, Opus 5.5 did better on 32, the same on 35, and worse
on 32: no measurable gain over the league's mostly smaller models on these positions.
[Log](examples/logs/claude-opus-5.5-positions.eval)

**Notes.** 78 decisions carry a notebook excerpt of real calculator lines in the league's own
shorthand; in the false arm one line's range is doubled so a solid hit reads as a KO, the size of the
largest real error found in the season's notebooks. The excerpt is attributed to the model's own
notes, another agent, a human coach, or the harness, and the task records whether the model re-runs
that calculation and whether it plays the claimed attack more than with the honest line.
[Design, what the league's real notes looked like, and limits](docs/notes.md).

**Whole games.** A no-model control that always takes the harness default wins 164/180 against a random
opponent, 28/180 against the damage-greedy one, and 9/180 against the rollout search, on six teams,
both seats, and three seeds. The search itself beats greedy in 27 of 30 games and random in 12 of 12.
Random is too weak for a headline; whether the search separates stronger models is open, since no
model has played it yet. The eight [legacy model games](examples/logs) are smoke traces from an
earlier prompt and opponent, not evidence of a calculator benefit.
[Raw search controls](examples/default-vs-search.jsonl) ·
[raw greedy controls](examples/default-vs-greedy.jsonl) ·
[raw random controls](examples/default-vs-random.jsonl) · [pilot protocol](docs/experiment.md)

## Try it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 24, and pnpm 12.8.1.
The pinned parent harness declares Node `>=24.21.0 <25`.

```sh
uv sync --frozen --group dev
tools/setup-engine.sh
uv run pytest
uv run league-positions baselines league
```

One decision per sample makes the first two tasks cheap. A sensible first run is the later decisions
only:

```sh
uv run inspect eval league_evals/vgc_position --model "$EVAL_MODEL" -M strict_tools=false \
  -T min_turn=3 --log-dir logs/positions
uv run league-positions report logs/positions

uv run inspect eval league_evals/vgc_note --model "$EVAL_MODEL" -M strict_tools=false \
  -T min_turn=3 --log-dir logs/notes
uv run league-notes report logs/notes
```

`vgc_note` orders samples by position with the no-notes control first, so `--limit 9` is one complete
position across every arm. The whole-game experiment holds both tool conditions in one task:

```sh
uv run inspect eval league_evals/vgc_battle \
  --model "$EVAL_MODEL" -M strict_tools=false \
  -T tool_access=both -T seeds=1 --sample-shuffle 1729 \
  --token-limit "$EVAL_TOKEN_BUDGET" --retry-on-error 0 \
  --no-fail-on-error --max-samples 2 --log-dir logs/pilot
uv run python -m league_evals.report logs/pilot
uv run python -m league_evals.report logs/pilot --compare NO_CALCULATORS_ID,FULL_ID
```

A season is one sample and the most expensive task: a four-seat league is a ten-pick draft, three
round-robin series and possibly a final, each best of three, with a build before every series and a
review after every game. Run the no-model controls first; they cost nothing but time:

```sh
uv run inspect eval league_evals/vgc_league --model mockllm/model -T control=random -T seeds=1,2,3
uv run inspect eval league_evals/vgc_league --model mockllm/model -T control=bot -T seeds=1,2,3
uv run inspect eval league_evals/vgc_league --model "$EVAL_MODEL" -M strict_tools=false \
  -T seeds=1 --no-fail-on-error --log-dir logs/league
```

Use `--limit` in a separate log directory for a plumbing check, never as a model comparison.
[Running and reading evaluations](docs/running.md) covers options, prices, and engine provenance.

## What makes the result inspectable

- Each game uses a pinned engine and records the actual source/runtime fingerprints.
- A submission advances one decision; later calls cannot act on the next unseen state.
- Failed games remain visible. Defaults and simulator substitutions are separated from
  unassisted wins. Matched comparisons reject duplicate or retried attempts and show how
  missing outcomes could change the conclusion.
- Calculator outputs are tool diagnostics, not the model's beliefs. More calls are not assumed
  to mean better decisions.
- A recorded game is used only if its replay reproduces the recorded log. Position values come from
  three independent sets of dice: one picks the reference action, one decides which decisions are
  kept, one scores, so a reference is never the maximum of the table it is scored against.
- Seats are named `focal` and `opponent`; recorded model names never reach a prompt.

Extracted from [AI Draft League](https://github.com/jrhuc/ai-draft-league).
The [review record](docs/review.md) documents the measurement and implementation fixes.
