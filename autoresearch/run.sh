#!/usr/bin/env bash
# Auto Data Pruning — Experiment Runner
#
# Usage:
#   ./run.sh              Run a single experiment (prune + train + eval)
#   ./run.sh --prepare    Only prepare data (download + tokenize)
#   ./run.sh --loop N     Run N experiments with keep-or-revert logic
#
# The agent edits prune.py, then runs this script to see val_bpb.

set -euo pipefail
cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo -e "${GREEN}[autoresearch]${NC} $*"; }
warn() { echo -e "${YELLOW}[autoresearch]${NC} $*"; }
err() { echo -e "${RED}[autoresearch]${NC} $*" >&2; }

get_val_bpb() {
    # Extract the last val_bpb from results.json
    python3 -c "
import json
with open('results.json') as f:
    results = json.load(f)
print(results[-1]['val_bpb'])
" 2>/dev/null || echo "inf"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
prepare() {
    log "Preparing data..."
    python3 prepare.py
    log "Data preparation complete."
}

run_experiment() {
    log "Starting experiment..."
    log "Pruning strategy from prune.py:"
    grep -E "^(SCORER_TYPE|PRUNER_TYPE|PRUNE_RATIO)" prune.py || true
    echo ""

    python3 train.py

    local bpb
    bpb=$(get_val_bpb)
    log "Experiment complete. val_bpb = $bpb"
}

run_loop() {
    local n_experiments=${1:-10}
    local best_bpb="inf"

    log "Running $n_experiments experiments with keep-or-revert..."

    # Ensure we have a clean git state
    if ! git diff --quiet prune.py 2>/dev/null; then
        warn "prune.py has uncommitted changes. Committing as baseline..."
        git add prune.py
        git commit -m "baseline prune.py" 2>/dev/null || true
    fi

    for i in $(seq 1 "$n_experiments"); do
        log "=== Experiment $i/$n_experiments ==="

        run_experiment
        local bpb
        bpb=$(get_val_bpb)

        # Compare with best
        local is_better
        is_better=$(python3 -c "print('yes' if float('$bpb') < float('$best_bpb') else 'no')")

        if [ "$is_better" = "yes" ]; then
            log "IMPROVED: $best_bpb → $bpb. Keeping changes."
            best_bpb=$bpb
            git add prune.py results.json
            git commit -m "improvement: val_bpb=$bpb (experiment $i)"
        else
            warn "No improvement ($bpb >= $best_bpb). Reverting prune.py."
            git checkout prune.py
        fi

        echo ""
    done

    log "Loop complete. Best val_bpb = $best_bpb"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-}" in
    --prepare)
        prepare
        ;;
    --loop)
        run_loop "${2:-10}"
        ;;
    *)
        run_experiment
        ;;
esac
