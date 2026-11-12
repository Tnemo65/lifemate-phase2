---
name: experiment
description: Take a LifeMate Agent idea or completed Run through the Hypothesis diary to an official verdict.
argument-hint: <approach> <idea>
---

# Experiment

Arguments are free text: the first word is the approach (`aNN_name`), the rest is the idea.
Use `uv run lifemate` for platform commands below. Invocation authorizes the normal diary writes,
implementation, Run submission and conclusion. Complete the workflow without confirmation. The only
question is the run-script fallback described below. Report concrete blockers instead of inventing results.

Use ASD-STE100-inspired simple English for every diary entry and final report. Use familiar words,
active voice and one idea per sentence. Aim for at most 20 words per sentence. Define abbreviations.
Review and simplify your text before sending it. This is a house style, not a claim of ASD-STE100
certification. Every text field is ONE sentence with a hard character limit, and the platform rejects
longer or multi-sentence text (no ; joins). Put each fact in its own field; never repeat a slot inside
another. Main Idea: one sentence, at most 200 characters. Example: "Return no tool calls to measure the floor score."
Read diary data only with `agent-flow show <name-or-id> --json`; `agent-flow search <idea>` is the duplicate
check and returns JSON directly without a --json flag.
Read Run data with `run-folder inspect/find --json`. These commands own hashing, validation and scoring.
Normal source editing and git operations remain ordinary development work.

## 1. Resolve and check, before changing code

Resolve the Agent/config from the idea, otherwise `lifemate_agents.approaches.<approach>.agent:Agent` and
`configs/<approach>/base.toml` (then default.toml, then the sole/closest existing variant). Choose and report
reasonable defaults without asking. Inspect existing code and working changes; preserve unrelated edits.
Read `agent-flow show <approach> --json` and search the idea. Show matching records as JSON to assess them.
If the same idea is in progress or concluded, stop with its ID/result, unless the user's idea explicitly says
retry. Claim refusal exit 3 also stops that duplicate; choose another idea only when orchestrating iterate.

First decide whether the current code ALREADY implements the requested idea: compare the selected code and
its actual change against the approach Flow, or use the user's explicit statement that the completed Run
contains the implementation. Unrelated working changes are not evidence that this idea is implemented.
A matching Run of the current baseline is not a Run of a NEW, unimplemented idea. In that case, register the
Hypothesis, implement it, fill inputs and make a new Run; never submit the baseline as the proposed change.
A merged-confirmation request is an intentional retry that MUST produce a fresh Run. Skip reuse for that
request, even if the code matches a worker Run. For ordinary experiments, before editing ANY source/config/run script, run:

```
uv run lifemate run-folder find --agent <module:Agent> --config <config> --split train --json
```

Exit 0 returns `.match.folder`; exit 1 is no reusable Run; exit 2 is an input error to diagnose.
The finder checks current source, config, weights and data. For a supplied folder outside the default runs directory, inspect it, then
verify current-input reuse with the same find command and `--runs-dir <absolute-supplied-folder>`. Inspection
alone proves artifact validity, not that it matches current code/config/data.
Only when the requested change is already implemented does a matching completed Run mean skip Implement AND Run, and leave all code untouched until after Conclusion.
Inspect/find use captured weight references on a submitting laptop; do not download GPU weights to inspect it.
Record any `.unverified_weights` in the report. If any Sample errored, stop before upload; abandon an already
registered experiment with the actual reason. Repairing an Agent and producing a new Run is a later attempt.

## 2. Register the Hypothesis

For a new approach, draft Main Idea and Flow, then create it with
`agent-flow create --name <approach> --main-idea '...' --flow <temporary-file>`.
Flow is Python-like, high-level pseudocode that a new reader understands in a few seconds. It must parse
as Python, have at most 15 lines (aim for 5 to 15) and at most 80 characters per line. Show only control flow:
branches, loops, decisions and tool calls. Leave out classes, imports, types and ordinary error handling.
Name steps as meaningful calls, such as reason(), execute() and validate(). Use # for short notes. Example:
```
def run(sample, tools):
    context = initialize(sample, tools)
    while not finished(context):
        action = reason(context)
        if action.is_tool_call:
            context.add(execute(action))
        else:
            return action.response
```
Keep temporary Flow files outside the source tree so they do not invalidate a completed Run.
Draft the Hypothesis slots from the idea and existing code diff. Each is one sentence; keep them separate:
- title: a noun phrase of at most 50 characters, e.g. "Validate calls before execution"
- change: what you change, at most 120 characters, e.g. "Validate each tool call against its schema first."
- mechanism: why it helps, at most 160 characters, e.g. "Invalid calls fail the whole Sample, so blocking them lifts rejected_calls."
- metric: F1 or one of timeout_exception, rejected_calls, empty_mismatch, ask, hash_mismatch, f1_below_1.
  The hypothesis is that this bucket loses fewer points.
- guardrail: a different one of those, F1, or latency, which must not get worse.
- evidence (optional): counts that support it, at most 160 characters, e.g. "40 of 120 failed Samples have a rejected call."
- predicted (optional): the expected mean train score.
Register BEFORE implementation, without asking:

```
uv run lifemate agent-flow claim <approach> --title '...' --change '...' --metric rejected_calls \
  --mechanism '...' --guardrail latency [--evidence '...'] [--predicted 12.5] [--sample <id> ...]
```

Save the returned `.experiment.id`. Evidence Samples come from the idea/Traces, never invented IDs.
An iterate worker also passes the orchestrator's exact `--baseline <token> --defer-baseline`; a merged
confirmation passes the same token WITHOUT defer. These are comparison metadata, not Hypothesis slots.
Show the registered Hypothesis in the final report; corrections use `agent-flow touch`, not a confirmation gate.

## 3. Implement and fill Run inputs

When no matching completed Run exists, skip implementation if the code already implements the idea/differs
from the approach Flow in that way. Otherwise implement the stated change. Write the experiment's FULL Flow:
copy the based_on Flow (`.based_on` in `agent-flow show <experiment> --json`, else the approach Flow), keep its
structure and names, and change only the lines this experiment changes. Attach it with
`agent-flow touch <experiment> --flow <file>`. The platform shows the diff. A kept or reverted experiment must
change at least one Flow line; only a baseline rerun keeps the Flow unchanged.
For a completed Run, attach that implemented Flow without editing its source.

Before a NEW Run, fill every missing Run input you can: wrap untraced model calls with lifemate.llm or
span(kind='model'), instrument meaningful steps, and add missing config and pinned weight references.
OpenAI SDK calls already receive automatic model traces unless disabled. Keep these tracing/provenance
repairs out of the hypothesis's Change diff and list them separately. Use deterministic decoding.
For a REUSED Run, postpone all these repairs until AFTER its submission and Conclusion.

## 4. Run or reuse

For a new Run, call `scripts/run.sh <agent> <config> train`. Its final line must be an existing local Run folder.
If a completed Run reports Sample errors, inspect that folder and abandon before upload; this is not a
launcher-repair/retry opportunity. Otherwise, if the script is missing, fails, or gives no existing folder, try ONCE to fix an obvious script error (path,
argument, simple invocation). Do not silently change providers, start servers, or run on a different machine.
If it still cannot run, ask the user to run the Agent and supply the Run folder path; continue from Submit
when it arrives. This is the only question in experiment. If work is abandoned, conclude abandoned with why.

For either path, `run-folder inspect <folder> --json --require-clean` must succeed before upload. Exit 1
means Sample errors: stop upload and conclude abandoned with their IDs/reason. Exit 2 means invalid artifacts:
report the diagnostic and abandon this attempt. Never manufacture provenance, scores or missing Traces.

## 5. Submit, evaluate, conclude

```
uv run lifemate submit <folder> --experiment <experiment>
uv run lifemate agent-flow decide <experiment>
```

Save the Run ID and absolute URL from submit. Decide exit 0 is keep, 1 revert, 4 means rerun baseline on the
reported data version, and 2 is an input/auth/network problem. Read official results and Traces with
`agent-flow show <run-id> --json` and `agent-flow show <run-id> --sample <id> --json`. Read the baseline Run
from the remembered decision/baseline cohort the same way. Identify changed buckets and regressed Samples
from those official records; do not reimplement scoring. Small-delta/latency warnings never change the verdict.
On exit 4, rerun the actual baseline source/config at that version. If the baseline has an experiment,
submit the rerun to that experiment. If its experiment_id is null, claim a new baseline-rerun experiment,
submit the unchanged baseline Run to it, and call `agent-flow establish-baseline <experiment>`.
This command checks the source, Agent, config and weights against the existing baseline. It establishes
only a missing version baseline. Each approach keeps a separate baseline for each data version.
A frozen parallel round must instead restart with the new automatic token. If that cannot be completed,
record an abandoned attempt with the actual blocker, not a guessed keep/revert.

Conclude with the server verdict, 1 to 4 findings, a reason and a next step:
`agent-flow conclude <experiment> --verdict kept|reverted --finding '...' [--finding '...'] --reason '...'
--next-step '...'`.
The platform records scores, deltas, points lost per bucket, Run links, errors, timing and whether the
prediction held. Do not repeat those numbers. A finding is one observation the platform cannot see, at most
140 characters, e.g. "63 ask Samples copy the right ask but miss earlier write calls (train_0025)."
Reason says why the hypothesis held or failed, in one sentence of at most 160 characters. Next step names one
action in at most 140 characters, or "No further work".
If a comparison is blocked, use `--verdict abandoned --execution-status blocked --blocker '...'
--reason '...' --next-step '...'`. The hypothesis is then inconclusive.
For work stopped by choice, use abandoned with execution-status stopped. Never leave claimed work dangling.
Reverse only your own candidate edits after a revert; preserve pre-existing user changes. Iterate workers
commit their kept code on their own branch and return that branch, experiment ID and decision to the parent.

Only AFTER concluding a reused Run, add its missing tracing/provenance inputs for the NEXT Run, without
altering the archived Run folder. Preserve those separately identified repairs if reverting your own change.
Report: recorded Hypothesis, official score and delta, verdict/reason, platform prediction result, regressions/Sample IDs,
Run link, untraced time, warnings/unverified weights, skipped Implement/Run steps, and every input/tracing
repair (including deferred repairs). A blocker report says exactly which steps/results are still missing.
