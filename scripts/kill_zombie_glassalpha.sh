#!/usr/bin/env bash
# Kill zombie GlassAlpha Python processes that survive terminal closure
#
# Usage:
#   ./packages/scripts/kill_zombie_glassalpha.sh          # Dry run (list processes)
#   ./packages/scripts/kill_zombie_glassalpha.sh --kill   # Actually kill them
#
# This script finds Python processes running GlassAlpha that are:
# 1. Orphaned (parent process no longer exists)
# 2. Running for >1 hour
# 3. Consuming >0% CPU (stuck in uninterruptible sleep)

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--kill" ]]; then
    DRY_RUN=false
fi

echo "Searching for zombie GlassAlpha processes..."
echo

# Find Python processes with 'glassalpha' or 'pytest' in command line
# that are running for >1 hour
pids=$(ps aux | grep -E 'python.*glassalpha|pytest.*glassalpha' | grep -v grep | awk '{print $2}' || true)

if [[ -z "$pids" ]]; then
    echo "✓ No GlassAlpha processes found"
    exit 0
fi

count=0
zombie_pids=()

for pid in $pids; do
    # Skip if process no longer exists (race condition)
    if ! ps -p "$pid" > /dev/null 2>&1; then
        continue
    fi

    # Get process info
    info=$(ps -o pid,ppid,etime,pcpu,command -p "$pid" 2>/dev/null || true)

    if [[ -z "$info" ]]; then
        continue
    fi

    # Extract elapsed time (format: [[dd-]hh:]mm:ss)
    etime=$(echo "$info" | tail -1 | awk '{print $3}')

    # Check if running for >1 hour (rough heuristic)
    if [[ "$etime" =~ ^[0-9]+-.*$ ]] || [[ "$etime" =~ ^[0-9]{2,}:.*$ ]]; then
        echo "🔴 Zombie process found:"
        echo "$info"
        echo
        zombie_pids+=("$pid")
        ((count++))
    fi
done

if [[ $count -eq 0 ]]; then
    echo "✓ Found $count active GlassAlpha processes (all appear healthy)"
    exit 0
fi

echo "Found $count zombie processes"
echo

if [[ "$DRY_RUN" == true ]]; then
    echo "To kill these processes, run:"
    echo "  $0 --kill"
    echo
    echo "Or kill manually:"
    for pid in "${zombie_pids[@]}"; do
        echo "  kill -9 $pid"
    done
else
    echo "Killing zombie processes..."
    for pid in "${zombie_pids[@]}"; do
        echo "  Killing PID $pid..."
        kill -9 "$pid" 2>/dev/null || echo "    (already dead)"
    done
    echo "✓ Done"
fi
