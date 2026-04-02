#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HISTORY_FILE="$REPO_ROOT/benches/history.csv"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
VERSION="$(cat "$REPO_ROOT/VERSION" 2>/dev/null || echo 'unknown')"

if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,git_sha,version,benchmark,low,estimate,high,unit" > "$HISTORY_FILE"
fi

echo "Running benchmarks (v${VERSION} @ ${GIT_SHA})..."
echo ""

BENCH_OUTPUT=$(cargo bench --all-features 2>&1)
echo "$BENCH_OUTPUT"

# Criterion outputs bench name on a line by itself or prefixed before "time:"
# Format:  "bench_name" on one line, then "  time:   [1.234 ns 5.678 ns 9.012 ns]"
# Or:      "bench_name        time:   [1.234 ns 5.678 ns 9.012 ns]"
BENCH_COUNT=0
PREV_NAME=""

while IFS= read -r line; do
    if echo "$line" | grep -qP 'time:\s+\['; then
        # Extract name if on same line
        NAME=$(echo "$line" | perl -pe 's/\s*time:.*//; s/^\s+//; s/\s+$//')
        [ -z "$NAME" ] && NAME="$PREV_NAME"

        # Extract [low unit est unit high unit]
        INNER=$(echo "$line" | perl -pe 's/.*\[//; s/\].*//')
        LOW=$(echo "$INNER" | awk '{print $1}')
        UNIT=$(echo "$INNER" | awk '{print $2}')
        EST=$(echo "$INNER" | awk '{print $3}')
        HIGH=$(echo "$INNER" | awk '{print $5}')

        if [ -n "$NAME" ] && [ -n "$EST" ]; then
            echo "${TIMESTAMP},${GIT_SHA},${VERSION},${NAME},${LOW},${EST},${HIGH},${UNIT}" >> "$HISTORY_FILE"
            BENCH_COUNT=$((BENCH_COUNT + 1))
        fi
    else
        STRIPPED=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [ -n "$STRIPPED" ] && ! echo "$STRIPPED" | grep -qE '^(Benchmarking|Running|Compiling|Compiled|Finished|Gnuplot|Found|change|warning|Warning|Performance)'; then
            PREV_NAME="$STRIPPED"
        fi
    fi
done <<< "$BENCH_OUTPUT"

echo ""
echo "Recorded ${BENCH_COUNT} benchmarks to benches/history.csv"
echo "Commit: ${GIT_SHA} | Version: ${VERSION} | Time: ${TIMESTAMP}"
