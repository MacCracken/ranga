# ADR 002: SemVer Versioning

**Status**: Accepted
**Date**: 2026-03-20

## Context

ranga is consumed by three downstream projects (rasa, tazama, aethersafta).
These projects need predictable API stability to avoid coordination overhead
when ranga changes.

## Decision

Follow Semantic Versioning strictly starting from v0.20.3:

- **Patch** (0.20.x): Bug fixes, documentation, infrastructure. No API changes.
- **Minor** (0.2x.0): New features, potentially breaking changes (allowed in 0.x).
- **Major** (1.0.0): Stable API. No breaking changes after 1.0 without major bump.

Enforce with `cargo-semver-checks` in CI on all pull requests.

## Consequences

### Positive

- Downstream projects can pin `~0.20` and receive only compatible updates
- Breaking changes are visible in version numbers
- Automated enforcement catches accidental API breakage

### Negative

- Must plan breaking changes deliberately (batch into minor releases)
- Version numbers advance faster during 0.x development phase

### Implementation

- CI runs `cargo semver-checks check-release` on PRs
- CHANGELOG.md documents all changes per version
- VERSION file, Cargo.toml, and git tags must stay in sync (enforced by
  release workflow)
