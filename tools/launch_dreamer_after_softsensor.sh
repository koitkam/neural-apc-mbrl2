#!/usr/bin/env bash
# Wait for the softsensor `workflow.runner` python process to exit, then
# launch the dreamer staged_runner. Designed to be invoked by a tmux
# session. The trailing `sleep infinity` keeps the tmux pane alive so
# logs are inspectable after dreamer exits.
#
# Detection logic:
#   We match the actual python interpreter (comm == "python"), with argv
#   containing both "-m workflow.runner" and "neural-softsensor-pytorch".
#   Matching by comm avoids false positives from the outer tmux/bash
#   wrappers, whose argv would otherwise also contain "workflow.runner".
set -u

LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/koitkam/neural-apc-pytorch/logs/staged_${LOG_TS}_dreamer_only.log"
mkdir -p "$(dirname "$LOG")"

_find_softsensor_pid() {
    # Print the PID of the softsensor python interpreter, or nothing.
    # Strategy:
    #   1) For each python process, split /proc/<pid>/cmdline on NUL and
    #      look for the exact tokens "-m" followed by "workflow.runner".
    #      This avoids matching dreamer's "workflow.staged_runner".
    #   2) Confirm cwd is inside neural-softsensor-pytorch so we don't
    #      pick up unrelated runners.
    local pid argv
    while read -r pid; do
        [ -z "$pid" ] && continue
        argv="$(tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null)"
        if printf '%s\n' "$argv" \
                | grep -A1 -Fx -- '-m' \
                | grep -Fx -- 'workflow.runner' >/dev/null \
            && readlink "/proc/${pid}/cwd" 2>/dev/null \
                | grep -q 'neural-softsensor-pytorch'; then
            echo "$pid"
            return
        fi
    done < <(pgrep -x python)
}

echo "[gate] $(date) waiting on softsensor python workflow.runner ..."
# Initial probe: if no softsensor is currently running, refuse to launch.
# This avoids the wrapper firing dreamer immediately when the user
# accidentally starts it before softsensor.
if [ -z "$(_find_softsensor_pid)" ]; then
    echo "[gate] $(date) no softsensor process detected. Aborting."
    echo "[gate] start softsensor first, then re-run this wrapper."
    sleep infinity
fi

while pid=$(_find_softsensor_pid); [ -n "$pid" ]; do
    sleep 30
done
echo "[gate] $(date) softsensor finished; launching dreamer."
echo "[gate] log -> ${LOG}"

cd /home/koitkam/neural-apc-pytorch || exit 2
# shellcheck disable=SC1091
source /home/koitkam/neural-apc-pytorch-env/bin/activate

python -u -m workflow.staged_runner \
    --simulation-dir simulation/test_sim \
    --agent-algo dreamer 2>&1 | tee "${LOG}"

EXIT=${PIPESTATUS[0]}
echo "=== dreamer finished $(date), exit code=${EXIT} ==="
echo "log: ${LOG}"
sleep infinity
