# Position evaluation

## Question

In decisions from recorded league games where the highest-damage joint action is not the
highest-win-rate joint action, which does a model choose?

The unit is a model-plus-tools system making one joint decision in the middle of a real game.
No game is played to the end and no judge model is involved. Each choice is scored against a table
the pinned simulator computed for that decision.

Damage is the natural foil. The harness's fixed opponent already plays the largest projected damage,
and a model that only ever does the same adds nothing over it. The decisions kept here are exactly
the ones where that policy gives up win rate: a Protect that denies a double target, speed control
before the attack, a switch that keeps a win condition alive, or an attack into the slot that matters
instead of the one that takes the most damage.

## Where the decisions come from

`league-positions build RUN_DIR` reads a finished AI Draft League run: both team
sheets, the simulator seed, every accepted choice of both seats, and the recorded log. The harness
replays each game from the seed and the choices and compares its log with the recorded one line by
line, wall-clock lines excluded. A game that does not reproduce is reported and contributes nothing.

Every simultaneous turn decision of both seats is a candidate. Team preview and forced switches are
not valued.

## How an action is valued

For one decision the harness takes every joint action the simulator accepts. It plays each against
two opponent replies: the damage-greedy reply and the reply the opponent actually made. After that
turn both sides continue with the damage-greedy policy to the end of the game, over reseeded dice.
An action's `value` is the focal seat's win rate over those rollouts. `explored` repeats this with
either side taking a random legal action a quarter of the time.

So a value is a win rate **conditional on that continuation**. It is not a game-theoretic value, and
it is not the win rate the model itself would go on to achieve.

## Selection

A decision has 50 to 400 joint actions, so the maximum of a noisy table is mostly noise. Selection is
built to keep that out of the scores.

1. A coarse pass values every action of every decision.
2. Each decision keeps a shortlist: its twelve best coarse actions, the greedy action, and the
   recorded action.
3. Three fine passes with independent dice value the shortlist: one chooses the reference, one
   decides which decisions are kept, one scores.
4. `best` is the top action of the choosing pass. `good` is every shortlist action within 0.10 of it
   in that pass.
5. The decision is kept when, in the keeping pass, `best` beats the greedy action by at least 0.20 in
   `value` and by at least 0.10 in `explored`.

Scores use the scoring pass only. The reference action and the decision to keep a position both come
from other dice, so the reference is never the maximum of the table it is scored against, the kept
gap is not the gap that is reported, and a model's regret can be negative.

The thresholds, sample counts, salts, and the counts at each stage are written into the dataset.
`--gap` and `--explored-gap` change the keeping rule; `--gap -1 --explored-gap -1` keeps every valued
decision, which is how `league-all` (494 decisions, the setting for [note trust](notes.md)) was built
from the same passes.

## Running it

```sh
uv run league-positions build /path/to/run --name league --jobs 8
uv run league-positions baselines league
uv run inspect eval league_evals/vgc_position --model PROVIDER/MODEL -T positions=league --log-dir logs/positions
uv run league-positions report logs/positions
```

The task fast-forwards the recorded game through the bridge, shows the model the decision exactly as
a league coach would see it with the game so far as its history, takes one submission, and stops.
The seats are named `focal` and `opponent`; the recorded model names never reach the prompt. The
model is told it is taking over a game under way and to maximise its probability of winning it.
`tool_access` works as in `vgc_battle`. `limit_hidden=0` keeps only decisions where the focal seat
had already seen every opposing Pokémon that was brought. `min_turn` drops earlier decisions.

An action outside the shortlist is valued when it is scored, with the dataset's own settings and
scoring salt. An action's value does not depend on which others are valued with it, so this equals
what a precomputed table would hold.

## Does the table mean anything?

The values assume a continuation no league model actually played, so they were checked against what
did happen. For all 494 valued decisions of the 27 verified games, the scoring-pass value of the action
the league model really took was compared with whether that model went on to win the real game.

| Decisions | n | AUC for the eventual winner | Mean value: eventual winners | Eventual losers |
| --- | --- | --- | --- | --- |
| All turns | 494 | 0.89 | 0.79 | 0.27 |
| Turns 1-2 | 108 | 0.68 | 0.67 | 0.47 |
| Turns 3-5 | 162 | 0.88 | 0.78 | 0.27 |
| Turn 6 on | 224 | 0.96 | 0.86 | 0.18 |

This shows the rollouts track how real games between non-greedy players ended. It mixes how good the
position was with how good the action was, so it is not evidence that any single action value is
right. It is weakest in the first two turns, where a plan's payoff depends most on later non-greedy
play: the largest recorded "miss" in the set is a turn-2 line from a game its player went on to win.
`min_turn=3` restricts a run to the later decisions; report it alongside the full set.

## Reading the scores

| Score | Meaning |
| --- | --- |
| `regret` | Scoring-pass value of `best` minus the value of the chosen action |
| `good` | The choice is within the good margin of `best` |
| `beats_greedy` | The choice has a higher value than the greedy action |
| `matches_recorded` | The choice is the action the league model played |
| `scored`, `on_demand`, `defaulted` | Whether a value exists, whether it was computed at scoring time, whether the harness played a default |

`baselines` prints the same scores at no model cost for the greedy action and the recorded league
decisions, overall and per recorded model. The uniformly random legal action has a mean regret only,
estimated from the coarse pass over every action.

## Limits

- **Selected on the foil.** Every kept decision is one where greedy loses win rate, so greedy's
  regret is large by construction, and a model that often plays the greedy action is penalised here
  by design. Mean regret on this set is not an estimate of general decision quality.
- **Recorded models are not compared like for like.** Each league model appears only in its own
  games, against its own opponents, with its own teams. The per-model baseline rows describe
  different decisions. A fresh model evaluated on all of them is comparable to another fresh model,
  and to the recorded decision on the same position, not to a row of the baseline table.
- **The continuation never switches, protects, or sets up by choice.** Lines whose payoff needs a
  later non-greedy move are undervalued. `explored` is a sensitivity check, not a second opinion from
  a stronger player.
- **Two opponent replies, not a best response.** A choice is not tested against the reply that
  punishes it most.
- **The simulator sees both teams.** Sheets are open, but which four were brought is not. A table
  can reward a line that was only right because of an unrevealed Pokémon. `hidden_opponents` records
  how many were unseen; report `limit_hidden=0` alongside the full set.
- **One decision out of its game.** The model sees the game so far but has none of its own earlier
  reasoning or notes, and is told it makes only this one decision.
- **Sampling noise.** Values are win rates over a finite number of rollouts. The scoring pass is
  independent of the pass that chose `best`, but each value still carries binomial error.
- **Same engine.** Values, tools, and legality come from one simulator. This is not an independent
  check of mechanics.
- **Small and public.** The decisions come from one season of eight models and are published with
  their values.

What the league shows beyond this, such as how notes drift between games, how a read of the opponent
hardens, and how a coach adapts its bring, stays qualitative and is written up outside this repository.
