# LifeMate phase 2: transcript to tool calls

Zalo AI Challenge 2026, LifeMate track. This repository holds our work on phase 2: a small model reads what the user said (as text) and calls the correct app tools in the organizers' simulator.

- Model: Qwen3-4B-Instruct-2507 with a LoRA adapter (the rules allow fewer than 5 billion parameters).
- Input: the correct transcript ("golden transcript"). Speech-to-text (phase 1) is not in this repository.
- Current score: **54.1 / 100** on our DEV split.

## Results

| Model | Train data | DEV (209 samples) |
|---|---|---|
| Base model, no training | none | 27.2 |
| `sft_v1`: LoRA, 2 epochs | 1,127 samples | 51.9 |
| `sft_v2`: `sft_v1` plus name masking on 30% of samples | 1,127 samples | 54.1 |
| `sft_lc50`: half of the train split, 4 epochs | 564 samples | 49.2 |

DEV is harder than train on purpose: 70% of its samples use 7 apps that the model never trains on. `sft_lc50` scores 67.3 on the 563 train samples that it did not see (`train_h50`).

A difference below about 3 points between two models is noise.

## System prompt

The system prompt is in [`p2/agent.py`](p2/agent.py), in the constant `SYSTEM_PROMPT` (line 43). Training and evaluation use the same text. It is in Vietnamese and has seven rules:

1. Call only the tools of the installed apps.
2. Compute relative times ("tomorrow", "next Friday") from `current_time`, and write them as ISO 8601 with `+07:00`.
3. Take IDs from the results of read tools. Do not invent them.
4. If a required value is missing, call `sys.ask_missing_info`, then stop.
5. If no installed app can do the request, give a short reply and call no tool.
6. Make no extra calls. Do not repeat the actions of earlier turns.
7. When the work is done, give a short reply with no tool call.

After the rules, the prompt adds the user profile, the current time, and a date table (today, tomorrow, and each weekday of this week and next week).

The model also gets the JSON schema of each installed tool (`tools_for`) and the earlier turns with their tool calls and results (`build_messages`).

`P2_RULES` (line 62) is an optional block of six more rules, taken from the reference answers. Switch it on with `--ctx p2`. It is off by default.

## Improvement ideas

[`p2/IDEAS.md`](p2/IDEAS.md) lists seven ideas with their evidence and a run order. Each idea is a switch that is off by default, so the default path is still the measured model. **No idea has a GPU result yet.**

| Switch | Idea |
|---|---|
| `--ctx p2` | Add the hidden habits of the reference answers to the system prompt |
| `--ctx kind` | Label each tool as read or write, and show what it returns |
| `--ctx compact` | Show tool results as numbered `key=value` lines |
| `--mask-epochs 4` | Pick the masked samples again in each epoch |
| `--boost llm:2` | Repeat conversation samples |
| `--base <name>` | Use another base model (not run) |
| in `synth_env.py` | Add missing trap types to the sample writer |

`python test_options.py` checks the switches on CPU.

## Layout

| Path | Content |
|---|---|
| `p2/agent.py` | System prompt, tool schema, and the loop: one tool call for each step, at most 10 steps |
| `p2/scorer.py` | Our copy of the official scoring formula |
| `p2/sft_data.py`, `p2/mask.py` | Build and check the training data; name masking |
| `p2/train_modal.py`, `p2/modal_app.py` | Train and evaluate on Modal |
| `p2/synth_env.py`, `p2/teacher_env.py`, `p2/guide_*.txt` | Write new training samples with two independent Claude runs |
| `p2/errors.py`, `p2/phenomena.py`, `p2/traps.py`, `p2/cmp_runs.py`, `p2/lc_report.py`, `p2/mine_rules.py` | Error analysis and run comparison |
| `p2/test_oracle.py`, `p2/test_options.py` | Local checks on CPU |
| `p2/IDEAS.md` | Improvement ideas: evidence, status, run order |
| `p2/artifacts/` | Split definition (`splits3.json`), hashes of the correct final states, mined answer conventions |
| `p2/runs/` | Score and training summaries of each run |
| `p2/MASTER_PLAN.md` | Full analysis and plan (Vietnamese) |
| `p2/V3_DESIGN.md`, `p2/RESEARCH_HYPOTHESES.md` | Design notes and paper review (Vietnamese) |
| `LIFEMATE_TONG_HOP.md` | Summary of the competition rules (Vietnamese) |
| `src/`, `configs/`, `scripts/`, `docs/` | LifeMate platform scaffold. It holds only the starter agent; our code is in `p2/`. |

## Splits

All splits come from the 1,545 labeled samples. They are fixed in `p2/artifacts/splits3.json`. Do not generate them again.

| Split | Samples | Use |
|---|---|---|
| `train` | 1,127 | Training |
| `dev` | 209 | Measure each change |
| `final_test` | 209 | Locked. Run once, after the method is final. |
| `train_q50` / `train_h50` | 564 / 563 | Two halves of `train` for experiments: train on one, measure on the other |
| `dev_clean` | 185 | `dev` without the 24 samples used to tune the sample writer |

## Setup

The competition data is not in this repository.

1. Put the organizers' data at `train+public_test/train+public_test/` in the repository root. It must contain `train/` and `public_test/`.
2. Put the tokenizer files of `Qwen/Qwen3-4B-Instruct-2507` in `p2/tokenizer/` (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`).
3. Install Python 3.11 or later with `transformers` and `modal`. Log in to Modal.

## Commands

Run all commands from `p2/`.

```
python test_oracle.py                         # local check: a perfect model must score 100
python sft_data.py --split train --mask 0.3   # build and check training data (CPU)
python -m modal run --detach train_modal.py --tag sft_v2 --splits train --mask 0.3
python -m modal run modal_app.py --split dev --model sft_v2 --tag sft_v2
python cmp_runs.py runs/sft_v1_B_24GB_safe_dev runs/sft_v2_dev
python errors.py runs/sft_v2_dev              # error types of one run
```

Training stops if the data built on Modal differs from the local build. Per-sample predictions are not in the repository, so `cmp_runs.py`, `errors.py` and `phenomena.py` need your own evaluation runs.

## Data rules

- Use the simulator only through `zaic_tools`. Do not read the `*_environment/` folders.
- Do not train on the public test set. Do not use prediction files as labels.
- Keep competition data, logs and per-sample outputs out of git. `.gitignore` already blocks them.
