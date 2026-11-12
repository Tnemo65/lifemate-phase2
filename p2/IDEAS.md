# Ideas to raise the phase 2 score

Date: 2026-11-12. Scope: golden transcript to tool calls, Qwen3-4B, LoRA.

**Status in one line:** every idea below is in the code as a switch that is OFF by default. No idea has a GPU result yet. The GPU account is over its spending limit.

## Rules for every idea

- One switch for one idea. The default path is the model that we measured (DEV 54.1).
- The default path did not change. Proof: `sft_data.py --split train_q50 --mask 0.3` gives the same sha256 as before (`4daeb65d…`).
- Train and eval must use the same variant. Training writes the variant to `train_info.json`. Eval reads it back (`--ctx auto`).
- Compare with `cmp_runs.py`. A difference below about 3 points is noise (teammate measured a 3-point move between two trainings of one recipe).
- Measure on `train_h50` (563 samples, model trained on `train_q50`) and confirm on `dev_clean`. Do not use `final_test`.

## Ideas that are in the code

| # | Idea | Switch | Evidence | Local check |
|---|---|---|---|---|
| 1 | Add the hidden habits of the reference answers to the system prompt | `--ctx p2` | With no rule, the model guesses instead of asking: 10 of 16 "read, then ask" samples on DEV, 15 of 63 on `train_h50`. The rule "1 match: act; 2 or more: ask" holds 255/255 and 65/65 in the reference answers. Teammate: ask accuracy fell from 62% to 50% when the prompt lost this rule; self-correction rose by 10 points with prompt rules. | Prompt grows by 777 characters. Rules do not name any app. |
| 2 | Label each tool as read or write, and show what it returns | `--ctx kind` | The catalog has `kind` and `returns`; the default prompt drops them. Teammate: flat alone (dev −0.4, held-out +0.2). | 49 of 49 app tools get a label. Labels stay after name masking. |
| 3 | Show tool results as numbered `key=value` lines | `--ctx compact` | Wrong item picked from a list is 11% of lost points on conversation samples. Teammate: pick-from-list 75% to 80%, platform +1.14 (inside noise on dev). | 5,054 tool results checked: no key or value is lost. Text is 83% of the JSON length. |
| 4 | Pick the masked samples again in each epoch | `--mask-epochs 4` | Today the same 30% of samples are always masked. On trained samples: masked 74.5, not masked 79.8. | Copy 0 equals the default data. With 4 copies, no sample is masked every time. |
| 5 | Repeat conversation samples | `--boost llm:2` | Conversation samples are 35% of train but the weakest group (40.2 against 72.7). Teammate repeats late samples twice, but only inside a combination. | Row count is correct. |
| 6 | Other base model | `--base <name>` | Teammate: Qwen3.5-4B gave the largest gain (platform +5.0, held-out apps +8.0). | **Not run.** Ask the teammate for the exact model ID. Library versions may need an update. |
| 7 | Add the missing trap types to the sample writer | on in `synth_env.py` | The pilot batch has too few of these: self-correction 0.5% (reference 3.9%, model score 35.5), spoken digits 0.5% (1.8%, 27.5), arithmetic 0.9% (3.4%), 3 or more apps 3.6% (12.8%), long sentences 1.8% (9.8%). | Code assigns the trap. Planned shares: 5.3 / 1.8 / 3.0 / 13.1 / 9.5%. The collector still rebuilds the 221 pilot samples exactly. No new samples written yet. |

With `p2+kind+compact`, the prompt is 5.5% longer (mean 5,993 tokens against 5,683). One of 564 samples then passes the 10,240-token training limit and is dropped.

## Ideas that we measured and did not build

| Idea | Why not |
|---|---|
| Guard: block a repeated identical call | The model repeats a call in 0 of 563 and 1 of 209 samples. No gain is possible. |
| Trap type "partial cancel" | The model scores 72.3 there, and the pilot batch already has 8.1% (reference 3.5%). |
| Guard: force "ask" after several matches | Teammate tried a code gate: platform −0.60, reverted. We use a prompt rule (idea 1). |
| Plan first, route first, batch calls | Teammate: all three lost 2.0 to 3.4 points. A 4B model does not recover from an early wrong decision. |
| Reasoning text, reinforcement learning | Xiaomi +0.3 and +0.6; Midea +3.1 with 96 GPUs for 95 hours. |

## Ideas that need GPU before code

- Rejection sampling (teammate T5): 6 rollouts for each train sample, keep the perfect ones. Dev +2.2, inside noise.
- Curriculum: written samples first, organizer samples last (Xiaomi: 94.6 against 93.0 mixed). Needs more than 221 written samples.

## Run order when GPU is back

Baseline for all comparisons: `sft_lc50` (train_q50, mask 0.3, 4 epochs; 67.3 on `train_h50`).

1. Finish the open work: score `sft_v2e4` on DEV; train and score `sft_lc50s` (221 written samples).
2. Zero-shot check of the prompt (no training, about 1 USD):
   `python -m modal run modal_app.py --split dev --ctx p2+kind --tag zs_p2kind`
   Compare with `runs/baseline_zeroshot_v2prompt_safe_dev` (27.2).
3. One training run with the three context ideas together (data file is ready):
   `python -m modal run --detach train_modal.py --tag sft_lc50c --splits train_q50 --mask 0.3 --epochs 4 --ctx p2+kind+compact`
   `python -m modal run modal_app.py --split train_h50 --model sft_lc50c --tag sft_lc50c`
   `python cmp_runs.py runs/sft_lc50_train_h50 runs/sft_lc50c_train_h50`
4. If step 3 wins, split it only if budget remains. If it loses, test `p2` alone.
5. Masking: `--mask-epochs 4` against `sft_lc50`. Also test `--mask 0`, because the teammate found that masking hurt held-out apps (−3.1).
6. Base model (idea 6) after the teammate gives the model ID.

Each training run on `train_q50` takes about 1 hour on one A100. Build the data file first with `sft_data.py` and the same options; training stops if the file does not match.

## Local checks

`python test_options.py` runs all checks on CPU. `python test_oracle.py` still gives 100.
