# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## Status: Rust line closed at 1.0.1 (2026-08-13)

1.0.0 shipped 2026-04-02 after the P(-1) scaffold hardening pass. **1.0.1 is the
final Rust release.** ranga continues in Cyrius, following mabda, prakash, and
ai-hwaccel, which have already ported.

What 1.0.1 covered:

- Toolchain current (verified on Rust 1.97.1), MSRV held at 1.89 so consumers can take it mid-port
- Dependencies refreshed; three RUSTSEC advisories cleared (0204 vulnerability, 0097 + 0190 unsoundness)
- crates.io publishing removed from the release pipeline — GitHub artifacts only

`wgpu` is frozen at 29 and `pollster` at 0.4 for the life of the Rust line, because
mabda 1.0.0 (the last Rust release of the AGNOS GPU foundation) links those majors.
See the note in `Cargo.toml` and the 1.0.1 CHANGELOG entry.

No further Rust feature work is planned. Bug fixes would only be cut if a consumer
hits a blocker before completing its own port.

---

## Carried to the Cyrius port

These were open against the Rust line and were never completed. They are recorded
here as input to the port, not as Rust commitments.

### API and correctness

- Document `Perspective` struct public fields (`a00`–`a12`)
- More granular error types — split the `RangaError::Other` catch-all where patterns emerge
- GPU `block_on` timeout mechanism — prevent theoretical infinite spin on GPU futures

### Testing

- Feature flag isolation tests — dedicated jobs per feature (`gpu`, `spectral`, `hwaccel`, `parallel`); the Rust CI only ever tested default, none, and all
- Feature interaction matrix — `gpu+hwaccel`, `parallel+simd`, `spectral+parallel`
- ICC `IccLutProfile` parsing coverage — minimal in the Rust implementation
- Fuzzing in CI — all 8 targets; `make fuzz` ran them locally but CI never gated on it
- GPU error propagation tests for chained dispatch failures
- Large-buffer stress tests (>1GB) for memory safety under pressure
- Visual regression baseline images (golden master approach)

### Benchmarks

- Missing coverage: `auto_white_balance`, `delta_e_cie94`, `fill_solid`, `nv12_to_rgba`
- GPU context creation overhead
- `hwaccel::probe()` and `should_use_gpu()` heuristics
- Benchmark regression detection — fail the build on >10% degradation against stored baseline
- Explicit parallel vs serial comparison
- GPU pipeline warm-up profiling — first-run vs cached dispatch cost

### Performance

- NEON (ARM) SIMD parity with SSE2/AVX2 — the Rust line hardened SSE2 in 1.0.0 but never verified NEON coverage matched

### Documentation

- Benchmark hardware specified in the performance table (CPU/GPU model)
- Explicit feature flag interaction documentation
- Consumer integration cookbook (rasa editor patterns, tazama video pipeline patterns)
- HWAccel decision-making heuristics guide

---

## Historical

`benches/history.csv` holds the full benchmark record across the Rust line
(0.20.3 → 1.0.1). Worth carrying forward as the port's performance target — the
Cyrius implementation should be measured against these numbers, not against a
fresh baseline.
