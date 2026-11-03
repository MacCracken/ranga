#!/usr/bin/env bash
set -euo pipefail

# Run ranga's Cyrius benchmark suite and append the results to a CSV history.
#
# Usage:
#   ./scripts/bench-history.sh                       # all suites -> bench-history.csv
#   ./scripts/bench-history.sh results.csv           # custom CSV
#   ./scripts/bench-history.sh "" tests/other.bcyr   # one custom suite
#
# The default suites are tests/ranga.bcyr (100 CPU benchmarks), tests/spectral.bcyr
# (8) and tests/gpu.bcyr (26). spectral needs prakash and gpu needs mabda, so each
# runs only when its library is vendored (`cyrius deps --features spectral,gpu`);
# otherwise it is skipped with a note. gpu.bcyr also needs an AMD GPU for its GPU
# half; without one it records only the CPU half. Its MISSING lines (operations
# the port does not have yet) carry no timing and are not recorded.
#
# Until 2.1.2 this was still the Rust script: it ran `cargo bench` and wrote
# benches/history.csv, neither of which exists in this tree. So it failed on every
# run, and no Cyrius benchmark history was ever recorded. It was rewritten during
# the 2.1.2 P(-1) pass. The frozen Rust series is docs/rust-v1-bench-history.csv.
#
# `cyrius bench` prints one line per benchmark:
#   <name>: <avg><unit> avg (min=<min><unit> max=<max><unit>) [<n> iters]
# Each row records avg and min in nanoseconds.
#
# Since cyrius 6.5.19 the harness calibrates one clock read and subtracts it from
# every sample, and it says so in a "[timer floor … subtracted from every sample]"
# line. `regime` is derived from that line rather than assumed. Rows from different
# regimes, or from different `cyrius` versions, are different instruments: compare
# within one only.

HISTORY_FILE="${1:-bench-history.csv}"
if [ -n "${2:-}" ]; then
    SUITES=("$2")
else
    SUITES=(tests/ranga.bcyr)
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
VERSION="$(tr -d '[:space:]' < VERSION)"
CYRIUS="$(grep -m1 '^cyrius = ' cyrius.cyml | sed 's/cyrius = "\(.*\)"/\1/')"

if [ -z "${2:-}" ]; then
    if [ -f lib/prakash.cyr ]; then
        SUITES+=(tests/spectral.bcyr)
    else
        echo "note: lib/prakash.cyr is not vendored; skipping tests/spectral.bcyr"
    fi
    if [ -f lib/mabda.cyr ]; then
        SUITES+=(tests/gpu.bcyr)
    else
        echo "note: lib/mabda.cyr is not vendored; skipping tests/gpu.bcyr"
    fi
fi

if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,commit,version,cyrius,benchmark,avg_ns,min_ns,iters,regime,floor_ns" > "$HISTORY_FILE"
fi

TOTAL=0
for SUITE in "${SUITES[@]}"; do
    echo "Running ${SUITE} (v${VERSION} @ ${COMMIT}, cyrius ${CYRIUS})..."
    OUT="$(cyrius bench "$SUITE" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')"
    echo "$OUT"
    echo ""

    # awk does the parsing: one pass, and the unit table lives in one place.
    COUNT="$(printf '%s\n' "$OUT" | awk -v ts="$TIMESTAMP" -v c="$COMMIT" -v v="$VERSION" \
        -v cy="$CYRIUS" -v csv="$HISTORY_FILE" '
        function ns(tok,   n, u) {
            n = tok; sub(/[a-zµ]+$/, "", n)
            u = tok; sub(/^[0-9.]+/, "", u)
            if (u == "ps") return n / 1000
            if (u == "ns") return n
            if (u == "us" || u == "µs") return n * 1000
            if (u == "ms") return n * 1000000
            if (u == "s")  return n * 1000000000
            return ""
        }
        /timer floor .* subtracted from every sample/ && floor == "" {
            match($0, /[0-9.]+(ps|ns|µs|us|ms|s) per clock read/)
            if (RSTART) { split(substr($0, RSTART, RLENGTH), f, " "); floor = ns(f[1]) }
        }
        / avg \(min=/ {
            line = $0; sub(/^[[:space:]]+/, "", line)
            name = line; sub(/:.*/, "", name)
            rest = line; sub(/^[^:]*:[[:space:]]*/, "", rest)
            split(rest, a, " ")
            mn = rest; sub(/.*min=/, "", mn); sub(/[[:space:]].*/, "", mn)
            it = rest; sub(/.*\[/, "", it); sub(/ iters.*/, "", it)
            rows[++nrows] = name "," ns(a[1]) "," ns(mn) "," it
        }
        END {
            regime = (floor == "") ? "floor_included" : "floor_subtracted"
            for (i = 1; i <= nrows; i++) {
                split(rows[i], r, ",")
                printf "%s,%s,%s,%s,%s,%.1f,%.1f,%s,%s,%s\n", ts, c, v, cy, r[1], r[2], r[3], r[4], regime, floor >> csv
            }
            print nrows
        }')"

    if [ "$COUNT" -eq 0 ]; then
        echo "ERROR: no benchmark rows parsed from ${SUITE}; stopping"
        exit 1
    fi
    TOTAL=$((TOTAL + COUNT))
done
echo "Recorded ${TOTAL} benchmarks to ${HISTORY_FILE} (v${VERSION} @ ${COMMIT}, cyrius ${CYRIUS})"
