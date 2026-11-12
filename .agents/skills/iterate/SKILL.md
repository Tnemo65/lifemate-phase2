---
name: iterate
description: Explore LifeMate Agent improvements in bounded rounds using parallel experiments and confirmed merges.
argument-hint: <approach> [parallel=3] [rounds=2] [min_samples=3]
---

# Iterate

Arguments are free text. First word: approach. Read parallel, rounds and min_samples from key=value or plain
words ("2 at a time, 5 rounds"). Defaults: parallel=3, rounds=2, min_samples=3; unreadable/non-positive values
fall back to their defaults. State the settings at the start and in the final report. Invocation authorizes
worktree branches/commits/merges and normal experiment diary writes, not pushes or provider provisioning.

Use the experiment skill for each idea. Read all diary/results/Traces through `agent-flow show ... --json`
(with --sample for a Trace), plus the explicit duplicate-search action. Platform commands use `uv run lifemate`.
Use existing scripts/run.sh; its one-repair-then-user-Run-path fallback is the only question. Worktree setup,
missing credentials or merge problems are concrete blockers to report without extra confirmation gates.

## Prepare

Read `agent-flow show <approach> --json` and its baseline Runs. If no baseline exists, use experiment once to
record the current implementation's initial train Run before planning a parallel round. Preserve existing
user edits: checkpoint/copy only the selected Agent/config/shared-code changes into an isolated integration
branch/worktree, without resetting or stashing unrelated work. Workers must receive the same current code,
config and baseline. A git HEAD that omits the working Agent changes is not an adequate round starting point.

## Each round (at most rounds)

1. Read show JSON again and freeze `.baseline.id` and its Run IDs for this whole round. Read their official
   bucket records/Traces via show JSON. Pick distinct ideas from the biggest fixable bucket(s) with at least
   min_samples real example IDs. Search for duplicates; stop if no eligible idea remains. Do not invent evidence.
2. Start up to parallel worker agents, each in its own branch/worktree FROM THE SAME round checkpoint.
   Give each the approach, one natural-language idea/evidence, Agent/config, and the exact baseline token.
   Each calls experiment, passes `--baseline <token> --defer-baseline` when claiming, and returns experiment ID,
   official verdict/delta, Run link, code commit/branch and any blockers. Deferred keeps record results without
   changing the approach baseline. Conclude stranded claims abandoned if their worker cannot finish.
3. If no worker is kept, stop. Merge only kept worker changes into a fresh confirmation worktree based on the
   round checkpoint. Resolve routine conflicts while preserving each change's intent; if changes cannot be
   combined, retain the single best kept branch instead of claiming an untested combination.
4. Register a NEW merged-confirmation experiment with the SAME `--baseline <token>` and WITHOUT defer. State
   "retry as merged confirmation" in its idea, even when only one worker was kept, so concluded-duplicate
   handling permits this deliberate confirmation. Describe the combination in the Hypothesis/Change. Use a
   fresh confirmation worktree with no copied worker Run artifacts, and explicitly require experiment to
   produce a NEW Run rather than reuse; this requirement overrides ordinary reuse. Run scripts/run.sh, inspect,
   submit, decide and conclude. Do not pass a worker's existing Run off as the merged confirmation.
   Only a kept confirmation may advance the baseline. A positive-looking sum of worker deltas is not a result.
5. If confirmation reverts/cannot complete, restore the single best kept worker's exact code/config. Adopt it
   only through `agent-flow conclude <kept-worker-experiment> --finalize`: this checks the deferred result
   against the current baseline before advancing it. If data/baseline changes make finalization fail, report
   the blocker and stop; do not set a baseline manually or falsify confirmation. Keep user edits intact.
6. Apply the confirmed/best adopted code to the integration branch, and check show JSON for the current
   automatic baseline before starting another round. Keep reverted worker branches out of the adopted code.

Stop at the round limit, a round with no keeps, no bucket with min_samples examples, or a concrete unrecoverable
blocker. Report settings, initial/final baseline score, each Hypothesis/verdict/delta/prediction, confirmation
and fallback-finalization outcomes, Run links, adopted branches, remaining buckets and a suggested next idea.
