# Changelog

## [0.20.5] — 2026-03-21

### Added

- **ARGB8 blend** — `blend_pixel_argb()` and `blend_row_normal_argb()` for ARGB channel layout (aethersafta native format)
- **ARGB8 positioned composite** — `composite_at_argb()` for ARGB8 layer compositing without RGBA conversion

### Changed

- Version bump for aethersafta ecosystem migration

## [0.20.4] — 2026-03-21

### Added

- **Geometry transforms** (`transform` module) — crop, resize (nearest/bilinear), affine transform, flip horizontal/vertical
- **Compositing** (`composite` module) — premultiplied alpha, layer masks, dissolve/fade/wipe transitions, solid/gradient/checkerboard fill, positioned composite with clipping
- **Filters** — median, bilateral, vibrance, channel mixer, threshold, flood fill (6 new, 23 total)
- **API improvements** — `Display` for `PixelFormat`/`BlendMode`/`ColorSpace`, `rows()`/`rows_mut()` iterators, `get_rgba()`/`set_rgba()` accessors, `from_view()`

### Changed

- `deny.toml` updated for cargo-deny v2 compatibility
- Coverage gate set to 75% in codecov.yml
- Documentation expanded: performance guide, testing guide, troubleshooting guide, comprehensive README

### Fixed

- Odd-dimension YUV/NV12 conversion OOB panic (chroma subsampling clamped)
- SIMD blend equivalence test tolerance for alpha rounding

## [0.20.3] — 2026-03-20

First release. Core image processing primitives for the AGNOS creative suite,
replacing inline implementations across rasa, tazama, and aethersafta.

### Color Science

- `LinRgba`, `Srgba`, `Hsl` types with bidirectional sRGB↔linear gamma conversion
- `CieXyz`, `CieLab` types with full sRGB↔XYZ↔Lab conversion chain (D65)
- `Cmyk` type with naive CMYK↔sRGB conversion
- Display P3 ↔ sRGB linear conversion (3x3 matrix)
- Color temperature (Kelvin → RGB multipliers, Tanner Helland approximation)
- Delta-E color distance: CIE76, CIE94, CIEDE2000 (Sharma 2005)
- `ColorSpace` enum: Srgb, LinearRgb, DisplayP3, Bt601, Bt709

### Pixel Buffer

- `PixelBuffer` with format validation, 6 formats (RGBA8, ARGB8, RGB8, YUV420p, NV12, RgbaF32)
- `PixelView` / `PixelViewMut` — zero-copy borrowed views for downstream integration
- `BufferPool` — reusable allocation pool for video frame pipelines

### Blend Modes

- 12 Porter-Duff blend modes: Normal, Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, SoftLight, HardLight, Difference, Exclusion
- SIMD acceleration: SSE2 (2px/iter), AVX2 (4px/iter, runtime detected), NEON (8px/iter)
- `blend_pixel` (single pixel) and `blend_row_normal` (row-level, SIMD-accelerated)

### Color Conversion

- RGBA↔YUV420p BT.601 (fixed-point)
- RGBA↔YUV420p BT.709 (fixed-point, HD video standard)
- ARGB→NV12, NV12→RGBA
- RGB8↔RGBA8, ARGB8↔RGBA8, RgbaF32↔RGBA8
- Odd-dimension safe (chroma subsampling clamped for non-even sizes)

### Filters (17 total)

- In-place: brightness, contrast, saturation, levels, curves, grayscale, invert
- Spatial: Gaussian blur (separable), box blur (separable), unsharp mask
- Color: hue shift (HSL), color balance (shadows/midtones/highlights)
- Effects: vignette, noise (Gaussian + salt-and-pepper with deterministic PRNG)
- Grading: 3D LUT application (.cube file parser with trilinear interpolation)
- Parallel blur via rayon (`parallel` feature)

### Histogram

- Luminance histogram (BT.601, configurable bins)
- Per-channel RGB histograms (256 bins, normalized)
- Chi-squared distance metric

### ICC Profiles

- Matrix-based ICC v2/v4 profile parser (pure Rust, no C deps)
- TRC support: gamma curves and lookup tables
- `IccProfile::apply()` for RGB→XYZ transform via parsed matrix + TRC

### GPU Compute (`gpu` feature)

- `GpuContext` — wgpu device/queue management (Vulkan + Metal)
- Pipeline caching — compiled shaders stored for reuse across calls
- WGSL compute shaders: blend (all 12 modes), invert, grayscale, brightness/contrast, saturation, Gaussian blur (horizontal + vertical)
- `GpuBuffer` — upload/download with async readback support
- GPU/CPU equivalence tests verify correctness within rounding tolerance

### Hardware Detection (`hwaccel` feature)

- `probe()` — GPU/Vulkan detection via ai-hwaccel
- `should_use_gpu(w, h)` — automatic CPU/GPU crossover recommendation

### Infrastructure

- CI/CD: 10-job GitHub Actions pipeline (lint, security, supply chain, test matrix, MSRV 1.89, coverage, benchmarks, docs, semver)
- Release workflow: tag-triggered 5-target build matrix, crates.io publish, GitHub Releases
- Supply chain: cargo-deny license allowlist, cargo-vet config
- 238 tests: 116 unit, 15 integration, 15 proptest, 92 doc-tests
- 37 criterion benchmarks across 6 suites (blend, convert, color science, filters, GPU)
- 3 fuzz targets (blend, convert, filter)
- 3 runnable examples
- 94.6% code coverage (75% CI gate)
- `#[non_exhaustive]` on all public enums
- All `unsafe` blocks documented with `// SAFETY:` comments
- SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, threat model, 2 ADRs, migration guide

### Feature Flags

- `simd` (default) — SSE2/AVX2/NEON blend acceleration
- `gpu` — wgpu compute shaders
- `hwaccel` — GPU detection via ai-hwaccel
- `parallel` — rayon row-parallel blur
- `full` — all features
