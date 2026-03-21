# ADR 004: Feature-Gated Optional Dependencies

**Status**: Accepted
**Date**: 2026-03-21

## Context

ranga has optional dependencies (wgpu, ai-hwaccel, rayon) that add significant
compile time and binary size. Not all consumers need GPU compute or hardware
detection.

## Decision

Each optional dependency is behind a feature flag:

| Feature | Dependency | Compile time impact |
|---------|-----------|-------------------|
| `simd` (default) | none | Zero — uses `std::arch` |
| `gpu` | wgpu (~200 transitive deps) | ~30s additional |
| `hwaccel` | ai-hwaccel (~5 deps) | ~2s additional |
| `parallel` | rayon (~10 deps) | ~3s additional |

The `full` feature enables everything. Default is `simd` only.

## Consequences

### Positive

- `cargo add ranga` compiles in ~3s with zero optional deps
- Consumers opt in to exactly what they need
- CI tests all feature combinations independently

### Negative

- Must maintain `#[cfg(feature = "...")]` gates throughout the code
- Feature interaction testing matrix grows (currently 5 combos tested)
- Documentation must clarify which features each function requires
