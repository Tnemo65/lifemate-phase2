#!/bin/sh
set -u
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 <agent> <config> [split]" >&2
    exit 2
fi
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || exit 1
cd "$project_root" || exit 1
agent=$1
config=$2
split=${3:-train}
log=$(mktemp "${TMPDIR:-/tmp}/lifemate-run.XXXXXX") || exit 1
trap 'rm -f "$log"' 0
if uv run lifemate run --agent "$agent" --config "$config" --split "$split" >"$log" 2>&1; then
    cat "$log"
else
    status=$?
    cat "$log" >&2
    exit "$status"
fi
run=$(sed -n 's/^run: //p' "$log" | tail -n 1)
if [ -z "$run" ] || [ ! -d "$run" ]; then
    echo "error: lifemate did not produce an existing Run folder" >&2
    exit 2
fi
run=$(CDPATH= cd -- "$run" && pwd -P) || exit 1
uv run lifemate run-folder inspect "$run" --json --require-clean || exit $?
printf '%s\n' "$run"
