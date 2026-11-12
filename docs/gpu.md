# Run on your compute

Use the same installed project on your laptop or GPU machine. On a new machine, `uv sync`, then copy the
WHOLE LifeMate data folder, including each hidden .lifemate-data.json marker. It is the project's data/ folder
(next to pyproject.toml); LIFEMATE_DATA_CACHE overrides its location. `lifemate data pull` on your logged-in laptop
gets data first. The GPU machine needs no LifeMate token. Keep weight references pinned in the config.

`scripts/run.sh <agent> <config> [split]` defaults to local execution. An adaptation must wait for completion,
copy back Submission/Traces/provenance/source, exit nonzero on failure, and print an existing LOCAL Run folder
as its final line. Inspect with `uv run lifemate run-folder inspect <folder> --json --require-clean` before submit.
Provider setup belongs to your launcher. The platform does not provision machines or select a model/provider.

## Modal example (comments to adapt)

Your own scripts/modal_run.py::main chooses compute, installs the pinned project, uses the copied cache,
runs lifemate, commits the Run volume, then returns its folder relative to that volume. No LifeMate credentials
belong in the Modal image or function. This example assumes your launcher already exists:

```sh
# result=$(mktemp)
# uv run modal run --write-result "$result" scripts/modal_run.py::main \
#   --agent "$agent" --config "$config" --split "$split" || exit $?
# remote_run=$(cat "$result") || exit $?
# local_run="$project_root/runs/$(basename "$remote_run")"
# mkdir -p "$local_run" || exit $?
# uv run modal volume get lifemate-runs "$remote_run" "$local_run" || exit $?
# uv run lifemate run-folder inspect "$local_run" --json --require-clean || exit $?
# printf '%s\n' "$local_run"
```

`modal run --write-result` writes a string/bytes return to a local file; volume get downloads directory
contents recursively. Persist writes with volume.commit() before returning. See [Modal run](https://modal.com/docs/cli/latest/run)
and [Modal Volumes](https://modal.com/docs/guide/volumes). Use your Modal profile and budget; these are examples,
not commands init runs.

## Vast or another SSH machine (comments to adapt)

Configure an SSH host alias gpu-box with the hostname/user/port/key from your rented instance's connection
details. Copy the project and data cache to /workspace/agent; do not copy Member credentials. With your existing
remote copy of the original local runner saved as scripts/run-local.sh before adapting your laptop runner:

```sh
# log=$(mktemp)
# ssh gpu-box "cd /workspace/agent && scripts/run-local.sh '$agent' '$config' '$split'" >"$log" || exit $?
# remote_run=$(tail -n 1 "$log")
# local_run="$project_root/runs/$(basename "$remote_run")"
# mkdir -p "$local_run" || exit $?
# rsync -a "gpu-box:$remote_run/" "$local_run/" || exit $?
# uv run lifemate run-folder inspect "$local_run" --json --require-clean || exit $?
# printf '%s\n' "$local_run"
```

Adapt quoting to your paths/launcher, retaining exit codes and local-path output. [Vast SSH connection details](https://docs.vast.ai/guides/instances/connect/ssh)
explain finding the instance-specific SSH command. Submit from your logged-in laptop with
`lifemate submit <folder> --experiment <id>`; keep/revert and Conclusions use the official replay score.
