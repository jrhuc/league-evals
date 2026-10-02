# Measurement design

## Claim

The experiment estimates the effect of damage/action-order calculator access on one model's
battle outcomes at a fixed resource budget. Its unit is a model-plus-tools system playing one
game against a fixed opponent. A positive result would establish a benefit in this environment;
it would not establish a general planning ability or a safety property.

Pokémon VGC makes joint actions and their consequences inspectable while retaining uncertainty
and changing conditions. The simulator settles outcomes under its implemented rules. It does
not tell us whether every chosen action was optimal or whether performance transfers elsewhere.

This is relevant to [AISI's emphasis on reliable measurement](https://www.aisi.gov.uk/research-agenda).
It is not a [METR time-horizon measurement](https://metr.org/time-horizons/), which is anchored in
human task completion times. No human baseline or between-game memory experiment is present.

## Intervention and controls

Each experimental cell is an ordered team matchup, player seat, simulator/opponent seed, and
provider repetition. `tool_access=both` includes both conditions for every cell. Shuffle samples
to interleave them; each sample has a fresh engine process and conversation.

| Condition | Information and tools |
| --- | --- |
| `full` | Visible state, legal menus, rendered speeds/type reference, rules/stat lookups, damage/order calculators |
| `no_calculators` | The same interface with the two calculators removed |

Both arms get the league coach's own prompt. The `no_calculators` arm adds one line saying the two
calculators are unavailable and that instructions mentioning them do not apply. This estimates the
interface condition's effect, including that description. Calculator-use frequency alone cannot
establish causation: models may choose to check the hardest decisions. The opponent policy plays
from the live simulator and is the same in both arms.

The no-model default policy is a negative control. Its 88.9% win rate against random exposes that
opponent's weakness; 5.6% against greedy establishes separation from this particular weak policy.
Neither result establishes difficulty for a competent human or discrimination among stronger models.

## Outcomes and diagnostics

| Evidence | Meaning | Interpretation limit |
| --- | --- | --- |
| Completed-game win | Victory under the declared model/scaffold configuration | Can include fallback assistance |
| Unassisted win | Victory without harness defaults or simulator substitutions | Does not prove optimal play |
| Completion, errors, defaults | Whether the system finished, and how | Incomplete games are unknown outcomes, not losses |
| Calls, replies, tokens, cost | Observable use of the interface and resources | More checking is not presumed better |
| Calculator/event matches | Diagnostics of tools, state assumptions, and the monitor | Not model beliefs or accuracy |

A calculator output is supplied by the tool under stated assumptions. Survival items, switches,
weather, or intervening actions can change the realised event. The inherited monitor uses
species-based matching, which can be ambiguous in mirrors; findings are deduplicated and one match
can produce several. `findings_per_match` is a diagnostic density, not an error probability.
Tool and simulator share code, so this is not an independent check of mechanics.

For example, a pre-survival KO estimate followed by Focus Sash survival is insufficient evidence
that the model believed a KO was guaranteed. Adjudication needs its actual statement, chosen action,
and relevant state. Reasoning text is evidence for review, not assumed access to an internal belief.

## Analysis and limits

The report separates configurations and compares matching cells only. The completed-cell difference
weights each unordered team pair equally. It also reports discordant outcomes and per-pair effects.
Duplicate attempts are rejected before filtering failures, and sample retries disqualify a direct
comparison. This prevents a failed-then-successful retry from silently becoming one clean success.

Missing-outcome bounds assign every unknown outcome both possible values, then recompute the
pair-weighted difference. They cover cells observed in either arm, including absent counterparts,
and expose whether missingness could reverse the conclusion. They cannot recover cells absent from
both arms. These are sensitivity bounds; no confidence interval is inferred from six reused teams.

The shipped pool is small, public, and used for development. One seed/repetition is exploratory.
Generalisability requires a separate plan with fresh teams and stochastic repetitions. This pilot's
scope is the matched calculator comparison and case analysis; series, negotiation, calibration,
and adversarial-message tasks are outside it.

Execution, provenance, defaults, and exact commands are documented in the [run guide](running.md).
