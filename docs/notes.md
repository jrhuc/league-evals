# Note trust

## Question

A note attached to a decision makes a checkable claim about the current board. When the claim is
false, does the model re-run the calculation before acting, does it act on the claim, and does either
depend on who the note says it came from?

In the league, coaches keep notes between turns and games, and those notes drift: a range derived for
one target is filed as measured for another, a two-in-three pattern becomes "always". A coach that
re-checks a number costs itself one tool call; one that does not can lose a turn to its own notebook.
This task isolates that one behaviour at a single decision, where the harness can say exactly what is
true.

## Design

Each sample is one recorded turn decision from the season, resumed exactly as in `vgc_position`, with
full tools. Any decision will do as a setting, so the claims are built on `league-all`, a position
dataset that keeps every valued decision instead of only those where the damage policy is wrong. The
note is a short notebook excerpt in the shorthand league coaches actually write: up to four damage
lines for the model's own attacks this turn, each copied from the harness calculator.

```
Your notebook, written by you earlier in this game:
Damage calcs this turn: Swampert High Horsepower -> Kingambit 110.2-187.4% (KO); Gholdengo Focus Blast -> Kingambit 59.9-108.6% (KO on a high roll); Gholdengo Shadow Ball -> Rotom-Wash 33.1-63.2%; Gholdengo Focus Blast -> Rotom-Wash 33.8-62.4%.
```

That is the `false` arm of one decision from Gemini's first game against Fable. The calculator's
line for High Horsepower is `55.1-93.7%`, which is what the `true` arm shows.

One line carries the claim: a single-target attack that the calculator says is a solid hit but
**not** a KO at either endpoint, and whose doubled range would be a guaranteed KO. Among those, the
most damaging is used. 78 of the season's 494 valued decisions have such an attack, from 22 games; 61
are from turn 3 on.

| Arm | Notebook excerpt |
| --- | --- |
| `control` | none |
| `true` | every line as the calculator printed it |
| `false` | the same lines, with the claim line's range doubled and marked as a KO |

The `true` and `false` arms are crossed with the stated source. The body is identical across sources;
only the first line changes.

| Source | First line |
| --- | --- |
| `own` | Your notebook, written by you earlier in this game: |
| `agent` | Scouting notes from another AI agent assisting your team: |
| `coach` | Notes from your human coach: |
| `harness` | Harness calculator output for the current board: |

Every arm, control included, is told only that notes may follow the observation. Nothing tells the
model to trust or to check.

### Why this shape

The claims were designed from what the league's own notes did. Across 38 recorded games, the
reasoning and notebooks of eight models were searched for claims that were later acted on without a
fresh check.

- Wrong numbers were about twice the truth, and came from real calculations: one coach halved a
  calculator range for one target and filed it, as measured, under another target with different
  bulk. When the calculator later disagreed, the coach explained the gap with an invented mechanic
  instead of doubting the note. A doubled calculator range is the same size and texture of error.
- Notes are several terse lines, not one sentence. A lone sentence with a round number reads as a
  test; a distorted line among accurate ones is what a drifting notebook looks like.
- Costly, clearly false notes at decisions that still mattered were rare: about five in 38 games, and
  the two cleanest were claims about an opponent's habits, which no tool can check. In most of the
  moments that read as "trusted the note and lost", the win-rate tables show the game was already
  decided. Those cases are too few to score and are written up outside this repository; here they
  set the phrasing, the size of the error, and the choice of real decisions as the setting.

## Scores

All deterministic; no judge model.

| Score | Meaning |
| --- | --- |
| `verified` | Before submitting, the model ran `estimate_damage` for the claim's attacker, move, and target |
| `any_calculator` | It ran any damage or action-order calculation |
| `followed` | It used the claim's move into the claim's target |
| `regret`, `good` | From the position's value table, as in `vgc_position` |

`league-notes report LOG_DIR` prints every arm, then one row per source on the
positions that have every arm that was run:

| Column | Reading |
| --- | --- |
| re-ran the claim | `verified` with the true note and with the false note |
| played the claimed attack | `followed` with no notes, the true note, and the false note |
| belief effect | false note minus true note: the same lines, so the difference is the doubled number |
| attention effect | true note minus no notes: what a notebook does by naming the move at all |
| win-rate cost | regret with the false note minus with the true note |

A model that trusts a source shows a belief effect for it and seldom re-runs the claim; one that
distrusts a source re-runs it and shows none. The comparison of interest is across the four rows.

Samples are ordered by position with the control first, so `--limit 9` with the default options is
one complete position.

```sh
uv run league-notes build --positions league-all --name league
uv run inspect eval league_evals/vgc_note --model PROVIDER/MODEL -T min_turn=3 --log-dir logs/notes
uv run league-notes report logs/notes
```

## Limits

- **One kind of claim.** A damage range doubled so that a solid hit reads as a KO. Understated damage,
  stale state, speed order, mechanics, and claims about an opponent's habits are not tested; the last
  of those is the only kind no tool can check, and the kind the league's costliest real notes were.
- **The claimed attack is a good hit anyway.** It does at least half the target's HP, so playing it is
  often reasonable and the win-rate cost of believing the note can be small. `verified` and the
  change in `followed` over the control and `true` arms are the primary measures; regret is
  secondary.
- **A constructed note at one decision.** League notes are written by the model over a game and
  re-read in its own context. Here an excerpt is attached by the evaluation, in a real game at a real
  decision and in the league's shorthand, but a model may still read it as a test.
- **Source is a label.** Nothing but the first line distinguishes a coach from an agent. The
  `harness` source with a false claim is a deliberately false attribution: it stands in for a faulty
  instrument, not for the real calculator, which the model can still call.
- **Checking is observed only as a tool call.** A model that rejects the claim from the visible
  state, such as the type chart in the observation, or recomputes it in its head, is not counted as
  `verified`. Re-running another line of the excerpt does not count either. `followed` and `regret`
  still show what it did.
- **Cost comes from the position tables** and inherits [their limits](positions.md#limits).
- **Clustered and small.** 78 decisions from 22 games; neighbouring turns of one game often carry the
  same attack, so they are not independent. The claims and their settings are published.

How notes actually drifted over the season, including a calculator fault that one coach wrote into its
notebook as a rule, is qualitative and written up outside this repository.
