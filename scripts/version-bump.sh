#!/usr/bin/env bash
set -euo pipefail

# ranga is a CYRIUS repo. This script used to be a Rust/Cargo script -- it sed'd
# Cargo.toml, ran `cargo generate-lockfile`, and verified a CARGO_VERSION. There is
# no Cargo.toml here, so it exited 1 on every invocation immediately after writing
# VERSION, leaving the dist bundles unregenerated. Rewritten 2026-09-11 during the
# cyrius 6.6.2 ecosystem sweep, which is where the failure surfaced.

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 2.0.2"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Bumping ranga to version ${NEW_VERSION}..."
echo "$NEW_VERSION" > VERSION
echo "  Updated VERSION"

# cyrius.cyml carries `version = "${file:VERSION}"`, so VERSION is the single source
# of truth and needs no second edit. The dist bundles DO carry a stamped version, and
# ranga ships FOUR of them -- the default plus three [lib.<profile>] bundles. Bare
# `cyrius distlib` regenerates only the default; a profile bundle left stale fails the
# version-consistency gate in CI.
cyrius distlib
for prof in $(grep -oE '^\[lib\.[a-z0-9_-]+\]' cyrius.cyml | sed 's/\[lib\.//;s/\]//'); do
    echo "  distlib profile: $prof"
    cyrius distlib "$prof"
done

FILE_VERSION="$(tr -d '[:space:]' < VERSION)"
if [ "$FILE_VERSION" != "$NEW_VERSION" ]; then
    echo "ERROR: VERSION is '$FILE_VERSION' after bump, expected '$NEW_VERSION'"
    exit 1
fi
for b in dist/*.cyr; do
    [ -f "$b" ] || continue
    stamp="$(grep -m1 '^# Version:' "$b" | sed 's/# Version: //')"
    if [ "$stamp" != "$NEW_VERSION" ]; then
        echo "ERROR: $b is stamped '$stamp', expected '$NEW_VERSION'"
        exit 1
    fi
done

echo ""
echo "Version bumped to ${NEW_VERSION}; all $(ls dist/*.cyr 2>/dev/null | wc -l) bundle(s) stamped."
echo "Next steps:"
echo "  git add -A && git commit -m \"bump to ${NEW_VERSION}\" && git tag ${NEW_VERSION}"
