# Changelog

## [0.21.3] — 2026-03-21

### Added

- **Oklab/Oklch color space** — `Oklab` and `Oklch` types with bidirectional conversion to/from linear sRGB (Björn Ottosson standard matrices)
- **BT.2020 color space** — `rgba_to_yuv420p_bt2020()` and `yuv420p_to_rgba_bt2020()` for UHD/HDR video wide-gamut conversion
- **Bicubic resize** — `ScaleFilter::Bicubic` variant using Catmull-Rom kernel for high-quality image scaling
- **Perspective transform** — `Perspective` struct with `from_quad()` 4-corner mapping, `perspective_transform()` function
- **Gradient radial fill** — `gradient_radial()` for center-outward radial gradients
- **Gradient angled fill** — `gradient_linear_angled()` for linear gradients at arbitrary angles
- **Histogram equalization** — `histogram::equalize()` for automatic contrast enhancement via CDF mapping
- **Auto-levels** — `histogram::auto_levels()` for linear min/max luminance stretching
- **Auto white balance** — `filter::auto_white_balance()` using gray-world algorithm
- **Embedded sRGB ICC profile** — `icc::srgb_v2_profile()` generates a minimal sRGB v2 ICC profile for embedding
- **ICC LUT-based profiles** — `IccLutProfile` struct with `from_bytes()` parser and `apply()` for mft1/mft2 tag types
- **GPU 3D LUT shader** — `LUT3D` WGSL compute shader with trilinear interpolation
- **GPU hue shift shader** — `HUE_SHIFT` WGSL compute shader (RGB→HSL→shift→RGB)
- **GPU color balance shader** — `COLOR_BALANCE` WGSL compute shader with shadow/midtone/highlight weighting
- **SIMD brightness filter** — SSE2 (x86_64) and NEON (aarch64) accelerated `brightness()` with saturating add/sub
- **SIMD grayscale filter** — SSE2 and NEON accelerated `grayscale()` using BT.601 coefficients
- **SIMD Y-plane conversion** — SSE2 and NEON accelerated luminance computation for BT.601/BT.709/BT.2020
- **Cache-aware blur tiling** — Vertical blur pass processes 64-pixel-wide strips for L2 cache locality
- **Photoshop reference test suite** — 12 blend mode golden-value tests verified against Photoshop output

### Changed

- `ColorSpace` enum now includes `Bt2020` variant
- Roadmap updated with completion status for all 0.21.3 items

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
