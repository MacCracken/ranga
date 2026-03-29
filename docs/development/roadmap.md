# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## Next Release: 1.0.0

### Pre-1.0 — Must Complete

#### API Hardening

- [ ] Add `#[must_use]` to all pure functions and functions returning values (~40+ missing across transform, convert, composite, filter, histogram, gpu, pixel modules)
- [ ] Document `Perspective` struct public fields (`a00`–`a12`)
- [ ] Final public API review — ensure every `pub` item has rustdoc with examples
- [ ] Run `cargo semver-checks` against 0.29.3 baseline before tagging 1.0

#### Testing

- [ ] Feature flag isolation tests in CI — dedicated jobs for `--features gpu`, `--features spectral`, `--features hwaccel`, `--features parallel` (currently only default, none, and all are tested)
- [ ] Feature interaction matrix — test `gpu+hwaccel`, `parallel+simd`, `spectral+parallel` combinations
- [ ] ICC `IccLutProfile` parsing coverage — currently minimal
- [ ] Fuzz testing in CI — add nightly job running all 8 fuzz targets (currently only 3 in `make fuzz`)
- [ ] Edge case tests for malformed/adversarial inputs on all conversion functions
- [ ] GPU error propagation tests for `GpuChain` chaining failures

#### Benchmarks

- [ ] Add missing benchmarks: `auto_white_balance`, `delta_e_cie94`, `fill_solid`, `nv12_to_rgba`
- [ ] Benchmark `GpuContext::new()` creation overhead
- [ ] Benchmark `hwaccel::probe()` and `should_use_gpu()` heuristics
- [ ] Baseline all benchmarks — final pre-1.0 `bench-history.sh` run

#### Documentation

- [ ] Specify benchmark hardware in README performance table (CPU/GPU model)
- [ ] Document feature flag interactions explicitly (which features compose, which are independent)
- [ ] MIGRATION.md — upgrade guide from 0.x to 1.0 for downstream consumers (rasa, tazama, aethersafta, soorat)

#### CI / Tooling

- [ ] Benchmark regression detection — compare against stored baseline, fail on >10% degradation
- [ ] Extend `make fuzz` to run all 8 fuzz targets (currently runs blend, convert, filter only)
- [ ] Add `make msrv` target for local MSRV verification

---

### Post-1.0 — Backlog

#### Features

- [ ] Additional examples: `gpu_blur.rs`, `crop_and_resize.rs`, `histogram_equalize.rs`, `spectral_analysis.rs`
- [ ] More granular error types — split `RangaError::Other` catch-all where patterns emerge
- [ ] GPU `block_on` timeout mechanism — prevent theoretical infinite spin on wgpu futures
- [ ] Async GPU readback patterns — benchmark and document async vs sync tradeoffs

#### Testing & Quality

- [ ] Large-buffer stress tests (>1GB) for memory safety under pressure
- [ ] Spectral + ICC interaction tests (cross-feature integration)
- [ ] Filter combination property tests (sequential filter application correctness)
- [ ] Doc-test coverage tracking and gating in CI
- [ ] Visual regression test baseline images (golden master approach)

#### Performance

- [ ] SIMD paths for NEON (ARM) — verify coverage parity with SSE2/AVX2
- [ ] Parallel (rayon) benchmarks — explicit parallel vs serial comparison
- [ ] GPU pipeline warm-up profiling — quantify first-run vs cached dispatch cost

#### Documentation

- [ ] Architecture decision records for 1.0 stability guarantees
- [ ] Consumer integration cookbook (rasa editor patterns, tazama video pipeline patterns)
- [ ] HWAccel decision-making heuristics guide
