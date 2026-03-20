# Changelog

## [0.20.3] — 2026-03-20

First release. Core image processing primitives extracted from rasa and aethersafta.

### Added

- **Color module** — `LinRgba`, `Srgba`, `Hsl` types with sRGB↔linear gamma conversion, HSL conversion
- **Pixel buffer** — `PixelBuffer` type with format validation, 6 pixel formats (RGBA8, ARGB8, RGB8, YUV420p, NV12, RgbaF32)
- **Blend modes** — 12 Porter-Duff blend modes (Normal, Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, SoftLight, HardLight, Difference, Exclusion)
- **Color conversion** — RGBA↔YUV420p (BT.601 fixed-point), ARGB→NV12
- **Filters** — brightness, contrast, saturation, levels, curves, grayscale, invert (all in-place on RGBA8)
- **Histogram** — luminance histogram, per-channel RGB histograms, chi-squared distance
- **CI/CD** — 10-job GitHub Actions pipeline (lint, security, supply chain, test matrix, MSRV, coverage, benchmarks, docs, semver)
- **Release workflow** — Tag-triggered build matrix with crates.io publish and GitHub Releases
- **Supply chain** — `deny.toml` with license allowlist, `supply-chain/` cargo-vet config
- **Testing** — 32 unit tests, 15 doc-tests, 15 integration tests, 3 fuzz targets, 3 examples
- **Documentation** — SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, threat model, ADRs, codecov config
