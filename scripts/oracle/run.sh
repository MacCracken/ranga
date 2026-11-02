#!/usr/bin/env bash
set -euo pipefail

# Differential oracle: build the frozen Rust 1.0.1 crate's driver and the Cyrius
# probe, run both on identical inputs, and diff them. See scripts/oracle/README.md.
#
# Usage:
#   ./scripts/oracle/run.sh                  # every group
#   ./scripts/oracle/run.sh levels affine    # only these groups
#   ./scripts/oracle/run.sh --detail levels  # extra args after groups go to compare.py
#
# Exits non-zero when any group differs from Rust without an accepted reason.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SELECT=()
CMP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --*) CMP_ARGS+=("$@"); break ;;
        *) SELECT+=("$1"); shift ;;
    esac
done

OUT="$REPO_ROOT/build/oracle"
mkdir -p "$OUT"
# build/ is gitignored, so the Rust build products stay out of the tree. Point
# CARGO_TARGET_DIR elsewhere to reuse an existing target directory.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$OUT/target}"

echo "building the Rust oracle (rust-old/ by path; rust-old/ itself is not modified)..."
cargo build --release --quiet --manifest-path scripts/oracle/rust/Cargo.toml
echo "building the Cyrius probe..."
cyrius build scripts/oracle/probe.cyr "$OUT/probe" > "$OUT/probe.build" 2>&1 \
    || { cat "$OUT/probe.build"; exit 1; }

"$CARGO_TARGET_DIR/release/ranga-oracle" "${SELECT[@]+"${SELECT[@]}"}" > "$OUT/rust.out"
"$OUT/probe" "${SELECT[@]+"${SELECT[@]}"}" > "$OUT/cyrius.out"
echo "rust: $(wc -l < "$OUT/rust.out") values; cyrius: $(wc -l < "$OUT/cyrius.out") values"
echo ""
python3 -I scripts/oracle/compare.py "$OUT/rust.out" "$OUT/cyrius.out" "${CMP_ARGS[@]+"${CMP_ARGS[@]}"}"
