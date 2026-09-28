# Review record — 2026-09-18

The original prototype provided a useful game interface, but its claims and reporting exceeded
what the recorded evidence could support. The revision centres on one question: does calculator
access improve matched battle outcomes?

## Findings and changes

| Finding | Resolution |
| --- | --- |
| Tool outputs described as model beliefs/calibration | Treat them as diagnostics; matched outcomes are the primary evidence |
| Engine lock said `unpinned`; bridge existed only in a sibling working tree | Pinned base plus checked-in patch, source/runtime hashes, and build manifest; sibling files preserved |
| Calls after submission could affect an unseen next decision | Sequential calls and a decision-closed guard, tested through Inspect |
| Closed-sheet setting did not prevent simulator disclosure | Unsupported setting rejected before generation |
| Failed attempts omitted; unfinished games could score as losses | Preserve failures and partial traces; unfinished outcomes remain unknown |
| Defaults could win without explicit assistance accounting | Separate assisted and unassisted wins |
| Grouping merged budgets, prompts, engines, and task versions | Configuration-specific reports and matched-cell comparisons |
| Focal always p1; limited examples overrepresented one team | Both seats and matchup directions, with the two tool conditions in one shuffled run |
| Failed-then-successful attempts could escape duplicate checks | Reject duplicates before dropping failures; reject sample retries in comparisons |
| Pair bootstrap suggested unsupported precision from six shared teams | Show outcomes and missing-outcome bounds; omit confidence intervals |
| Bridge reads could hang; schema errors were missed | Request bounds, response-ID checks, process cleanup, and Inspect-boundary error counts |
| Pricing assumed a cache discount and omitted cache writes | Recorded billing or explicit token-category prices; otherwise unknown |

## Evidence retained

The default policy completed 180 games against each opponent on the same six-team, three-seed,
both-seat matrix. It won 160 against random and 10 against greedy. All 360 games had zero simulator
rejections or substitutions, and their raw-log hashes verify. These are controls, not a model ranking
or a demonstrated calculator benefit.

The eight legacy model games remain unchanged as historical smoke evidence. Coverage and versions
differed, and one win used a harness fallback. The Focus Sash diagnostic does not independently
prove a model decision error. Their provenance lacks the runtime fingerprints added by this revision.

## Validation

The Python suite exercises the Inspect solver, both seats against the real engine, matched tool
conditions, failed attempts, cost accounting, provenance drift, and replay with wall-clock lines
removed. Lint, formatting, source-distribution/wheel builds, and installed pool loading are checked.

The documented engine setup also completed from a fresh network clone in an isolated temporary
directory, including dependency installation, Showdown build, bridge build, manifest validation,
and all nine engine integration tests. This adds a clean local build check; the Linux CI job has
not been run in this session.

## Remaining question

No paid model ablation has been run. The [120-game pilot](experiment.md) is the next empirical step.
Write up its complete matrix, costs, failures, and a few adjudicated cases. If both arms saturate,
report that limitation and stop this pilot. Scope expansion needs evidence and a separate plan.
