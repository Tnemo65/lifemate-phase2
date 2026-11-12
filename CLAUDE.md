# LifeMate Agent workspace

Setup: install the LifeMate Git dependency, `lifemate login`, then `lifemate init` (includes data pull).
Agents: `src/lifemate_agents/approaches/<aNN_name>/agent.py`, Agent(config)/setup/run; configs:
`configs/<aNN_name>/base.toml`. Use the env's public catalog/validate/call interface and deterministic decoding.

Use the experiment skill for an idea or completed Run; use iterate for bounded rounds. They record the
Hypothesis before implementation and finish with an official server score and Conclusion. Read diary data
with `uv run lifemate agent-flow show <name-or-id> --json`. Only total train scores decide; baselines are automatic.
Every Agent Run goes through `scripts/run.sh <module:Agent> <config> [split=train]`; its last line is the local
Run folder path. Edit that script once for your compute, following `docs/gpu.md`. No Sample errors may be uploaded.
Use `lifemate.llm`/spans for tracing and pinned weight references for reproducibility.
Use ASD-STE100-inspired simple English in every agent-written record and report: familiar words, active voice,
one idea per sentence, and short sentences. Review language before writing. Every diary text field is one
sentence with a hard character limit. Flow is Python-like pseudocode of at most 15 lines that parses as Python.
Keep hypothesis slots separate. Record findings, blockers and next steps in their own conclusion fields. Scores and timing come from platform evidence. Keep credentials out of source/config.
