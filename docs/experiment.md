# Pilot: does calculator access help?

**Question:** Does one fixed model win more often against greedy when given damage and
action-order calculators, under the same reply and token limits?

**Status:** protocol implemented; real-model comparison not yet run. The existing controls establish
that greedy distinguishes a weak default policy. They do not answer this question.

## One run, two conditions

`tool_access=both` creates each cell twice: `full` and `no_calculators`. Both keep the same
observation, legal menus, effective-speed display, rules/stat lookups, and conversation memory.
The league prompt is the same in both arms; the `no_calculators` arm adds one line saying the two
calculators are unavailable. The opponent policy is the same in both arms.
Each sample starts a fresh game and model conversation.

The first pilot is **120 games**: six teams × five opponents × two seats × one seed × two
conditions. Use `--sample-shuffle 1729` to mix both conditions in one evaluation rather than run
one arm systematically earlier. The shuffle seed fixes ordering; it does not seed model outputs.
This small development-pool pilot is for detecting useful signal and diagnosing failures.

Before running, record the exact provider/model, generation settings (including reasoning effort),
engine hashes, token limit, and sample count. Use a separate two-game smoke run to estimate cost;
then freeze the pilot's settings. The command is in the [README](../README.md). Do not use `--limit`
for the comparison, change budgets halfway through, or retry individual failures/losses.

Default settings are the `search` opponent (pass `-T opponent=greedy` for the question above),
both player seats, 24 replies per decision, and a 200-decision cutoff. Inspect sample retries are
disabled; provider HTTP retries may still occur.
The token limit counts the growing conversation and tool traffic. This tests the tool-access
condition at a resource budget, including the tool descriptions, rather than pure reasoning ability.

## Read the result

1. Confirm **60 retained records per condition, all expected cells, and matching engine/settings**.
   Report incomplete games and assistance first. The report keeps failures and rejects duplicate
   cells or sample retries in comparisons.
2. Compare `NO_CALCULATORS_ID,FULL_ID`. The estimate is full minus no-calculators on matched
   completed cells, averaged equally over unordered team pairs. The output also gives wins unique
   to each arm, unchanged outcomes, and per-pair differences.
3. Read `missing_outcome_bounds`: the worst/best difference if unfinished or missing games had
   lost/won. These cover cells observed in either arm; they cannot detect cells absent from both.
   They are sensitivity bounds, not confidence intervals. A favourable complete-case estimate
   alone is insufficient if missing outcomes could reverse it.
4. Inspect discordant games and all failures. Distinguish infrastructure, interface, tool/state
   assumptions, monitor issues, decision errors, and uncertain cases. Preserve the relevant
   observation, tool result, action, and simulator event. One unlucky loss is not a decision error.
5. Report tokens and cost alongside outcomes. State the fixed pool and seed explicitly. Shared
   teams and a single model repetition do not support generalisation or a model ranking.

## Stop and write it up

Publish the full matrix, logs, failures, cost, and a few adjudicated cases in a short result note.
A negative result is useful if it shows where checking adds no value or why the instrument fails.

If both arms saturate, treat this opponent/pool as insufficiently discriminating and stop the pilot.
If outcomes differ, inspect the cases before expanding. A follow-up with more seeds/repetitions or
held-out teams needs its own frozen plan. Neither outcome automatically justifies adding more tasks,
models, a leaderboard, or a broader safety claim.
