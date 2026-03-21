# Testing Guide

## Running Tests

```sh
# All tests with default features
cargo test

# All features (includes GPU tests — skipped if no GPU)
cargo test --all-features

# Specific module
cargo test -- color::tests
cargo test -- convert::tests

# Doc-tests only
cargo test --doc
```

## Test Matrix

| Feature combo | What it covers |
|---|---|
| `cargo test` | Core + SIMD blend + scalar filters |
| `cargo test --no-default-features` | Pure scalar, no SIMD |
| `cargo test --features gpu` | GPU shaders + CPU equivalence |
| `cargo test --features parallel` | Rayon blur paths |
| `cargo test --all-features` | Everything |

## Test Categories

- **Unit tests** (in-module `#[cfg(test)]`): 116 tests covering all functions
- **Integration tests** (`tests/integration.rs`): 15 end-to-end tests
- **Property tests** (`tests/proptest.rs`): 15 proptest properties (roundtrips, invariants)
- **Doc-tests**: 92 runnable examples in doc comments
- **GPU equivalence**: GPU output compared to CPU within +/-1 tolerance

## Coverage

```sh
cargo llvm-cov --html --output-dir coverage/
open coverage/html/index.html
```

Current coverage: 94.6%. CI gate: 75% (via codecov.yml).

## Fuzzing

Requires nightly and `cargo-fuzz`:

```sh
cargo +nightly fuzz run fuzz_blend -- -max_total_time=60
cargo +nightly fuzz run fuzz_convert -- -max_total_time=60
cargo +nightly fuzz run fuzz_filter -- -max_total_time=60
```

Fuzz targets are in `fuzz/fuzz_targets/`. They exercise:
- All 12 blend modes with arbitrary pixel data
- Format conversions with arbitrary buffer sizes
- All filters with arbitrary parameters

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) runs:
1. Format check (`cargo fmt`)
2. Clippy with `-D warnings` (3 feature variations)
3. Security audit (`cargo audit`)
4. Supply chain (`cargo deny`)
5. Test matrix (Linux, macOS, Windows)
6. Minimal features test (`--no-default-features`)
7. MSRV check (Rust 1.89)
8. Coverage upload to Codecov
9. Benchmark run (artifacts saved)
10. Doc build with `-D warnings`
11. SemVer check on PRs
