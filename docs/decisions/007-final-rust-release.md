# ADR 007: 1.0.1 Is the Final Rust Release

**Status**: Accepted
**Date**: 2026-08-13

## Context

The AGNOS stack is migrating to Cyrius. ranga's three AGNOS dependencies have
already ported — mabda, prakash, and ai-hwaccel each carry a `cyrius.cyml` and
their Rust crates.io versions are frozen artifacts (mabda 1.0.0, prakash 1.2.0,
ai-hwaccel 1.2.0). ranga's consumers (rasa, tazama, aethersafta, soorat) are
following.

This has a concrete effect on ranga's Rust line. ADR 005 accepted "ranga's GPU
layer is coupled to mabda's release cadence" as a tradeoff, on the reasoning that
mabda is an AGNOS crate under the same release process so version coordination is
straightforward. That coupling has now fully materialized: mabda 1.0.0 links
`wgpu ^29` and `pollster ^0.4`, and there will be no mabda 1.1 on crates.io.
wgpu 30 and pollster 1.0 exist upstream but are unreachable — adopting either
resolves a second copy of wgpu into the tree, and `mabda::GpuContext`'s types stop
unifying with ranga's.

So the Rust line cannot track its most significant dependency forward. Keeping it
open would mean either forking mabda's Rust implementation to chase wgpu, or
carrying a GPU stack that silently ages while the real work happens in Cyrius.

## Decision

**1.0.1 is the final Rust release of ranga.** Development continues in Cyrius.

- MSRV stays at **1.89** rather than advancing to current stable. Consumers still
  on the Rust line need to take 1.0.1 during their own ports; raising the floor
  would gate the last release behind a toolchain upgrade for no benefit.
- `wgpu` stays at **29** and `pollster` at **0.4**, frozen against mabda 1.0.0.
- crates.io publishing is removed from the release pipeline. 1.0.0 remains
  published; 1.0.1 ships as a git tag and GitHub release. The version-consistency
  gate that guarded the publish step is kept as a standalone `verify` job.
- No further feature work. A patch would only be cut if a consumer hits a blocker
  before completing its own port.

## Consequences

### Positive

- The Rust line closes in a known-good state: toolchain current as of 1.97.1, all
  RUSTSEC advisories cleared, full test suite and benchmark sweep green.
- No fork of mabda's Rust GPU layer, and no half-maintained wgpu upgrade path.
- `benches/history.csv` spans 0.20.3 → 1.0.1 and becomes the performance target
  the Cyrius implementation is measured against, rather than a fresh baseline.

### Negative

- Consumers that have not finished porting are pinned to wgpu 29 transitively
  through ranga. A consumer needing wgpu 30 for unrelated reasons has no path
  that keeps ranga in the tree.
- 1.0.1 is not installable from crates.io — consumers must take it as a git
  dependency on the tag, or stay on the published 1.0.0 and forgo the advisory
  fixes.
- Rust-side security advisories after this date go unaddressed. The frozen
  lockfile is clean as of 2026-08-13 and will drift.

### Mitigation

- The advisory fixes in 1.0.1 are the practical reason to take it; the git-tag
  dependency cost is one line in a consumer's manifest, and each consumer only
  carries it until its own port lands.
- The pinning rationale is recorded inline in `Cargo.toml` next to the `pollster`
  and `wgpu` entries, so anyone attempting a bump finds the reason at the point of
  the change rather than in the changelog.
