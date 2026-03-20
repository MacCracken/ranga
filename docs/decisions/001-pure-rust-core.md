# ADR 001: Pure Rust Core

**Status**: Accepted
**Date**: 2026-03-20

## Context

The AGNOS creative suite (rasa, tazama, aethersafta) needs a shared image
processing library for color conversion, blending, and filtering. Two
approaches were considered:

1. **C FFI wrappers** around optimized C libraries (libyuv, libblend)
2. **Pure Rust** implementation with SIMD via `std::arch` intrinsics

## Decision

Pure Rust core with no C FFI. SIMD acceleration uses `std::arch` intrinsics
behind `#[cfg(target_arch)]` gates. GPU compute uses wgpu (Rust bindings to
WebGPU).

## Consequences

### Positive

- Memory safety guaranteed by the compiler for all core operations
- Cross-platform without cross-compilation toolchain complexity
- Supply chain audit is simpler (no C dependencies to vet)
- Single build system (cargo) for all targets

### Negative

- Cannot leverage highly optimized C libraries (libyuv, Intel IPP)
- SIMD paths must be written and maintained per-architecture
- Initial performance may lag behind C equivalents

### Mitigation

- SIMD intrinsics for x86_64 (SSE2/AVX2) and aarch64 (NEON) close the
  performance gap on hot paths (blend, convert)
- Benchmarks track performance against known baselines
- GPU compute via wgpu offloads large-buffer operations entirely
