# Changelog

## [Unreleased] — 1.0.0

## [0.29.4] — 2026-03-29

### Changed

- **wgpu 24 → 29** — updated GPU compute backend to wgpu 29 (naga 29, wgpu-hal 29, wgpu-core 29)
  - Migrated `Maintain::Wait` → `PollType::wait_indefinitely()`
  - Migrated `Instance::new` to owned `InstanceDescriptor` via `new_without_display_handle()`
  - Migrated `request_adapter` from `Option` to `Result` return type
  - Migrated `PipelineLayoutDescriptor`: `push_constant_ranges` → `immediate_size`, `bind_group_layouts` now `Option`-wrapped
  - Removed extra `None` trace path argument from `request_device`
- **prakash 1.1.0 → 1.1.1** — bijli 1.0.1 dropped unnecessary soorat dependency; removed `default-features = false` workaround

## [0.29.3] — 2026-03-28

### Changed

- **Major dep upgrades**: ai-hwaccel 0.23.3→1.0.0, prakash 0.23.3→1.1.0 (hisab 0.22.4→1.3.0 transitive)
- Dependency update: wasm-bindgen 0.2.114→0.2.115, js-sys/web-sys 0.3.91→0.3.92, zerocopy 0.8.47→0.8.48
- Version bump from 0.24.3 to 0.29.3

### Added

- `#[must_use]` attributes on public functions in `composite` (11 functions), `pixel` (3 methods) — completing API hardening sweep
- Benchmarks: `auto_white_balance`, `delta_e_cie94`, `fill_solid`
- `Perspective` struct field documentation (3×3 homogeneous matrix element descriptions)
- `make msrv` target for local MSRV verification (`cargo +1.89 check --all-features`)
- 1.0.0 roadmap at `docs/development/roadmap.md` with pre-1.0 and post-1.0 backlog

### Fixed

- `make fuzz` now runs all 8 fuzz targets (was only 3: blend, convert, filter)
- Added `GPL-3.0-only` to `deny.toml` license allow list (hisab fixed SPDX identifier in 1.3.0)

## [0.24.3] — 2026-03-24

### Added

- **Spectral module** (`spectral` feature) — prakash integration for physically-based color science
  - `Spd` (spectral power distribution) type with CIE 1931 CMF integration
  - Bridged `From` conversions between `prakash::spectral::Xyz` and `ranga::color::CieXyz`
  - Convenience functions: `spd_to_xyz`, `xyz_to_cct`, `wavelength_to_xyz`, `d65_white`, `d50_white`, `blackbody_spd`
  - Re-exported standard illuminants (D65, D50, A, F2, F11), CIE 1931 2° CMFs, color rendering index
  - Re-exported `color_temperature_to_rgb`, `cct_from_xy` (inverse color temperature — new capability)
  - Re-exported high-precision sRGB gamma functions (`linear_to_srgb_gamma`, `srgb_gamma_to_linear`)
- **CieXyz white point constants** — `CieXyz::D65_WHITE` and `CieXyz::D50_WHITE` associated constants (always available, no feature gate)
- **ColorSpace::CieXyz variant** — added to the `ColorSpace` enum
- **Full test coverage sweep** — 37 new tests across 4 new test files (379 → 433 total)
  - `tests/edge_cases.rs` — 30 tests: error formatting, ARGB blend, composite_at_argb, histogram edge cases, filter edge cases, pixel edge cases, transform edge cases, convert edge cases
  - `tests/spectral.rs` — 17 tests: white points, XYZ roundtrip, SPD→XYZ, wavelength→XYZ, CCT, blackbody, CIE CMFs, illuminants, sRGB gamma, Wien peak, CRI
  - Expanded `tests/proptest.rs` — +7 property tests: Oklab/Oklch roundtrips, Delta-E CIE94, BT.2020 YUV, fade/wipe composites
- **Full benchmark coverage sweep** — 30 new benchmarks across 2 new + 4 expanded suites (~70 → ~108 total)
  - `benches/histogram.rs` (new) — luminance_histogram, rgb_histograms, equalize, auto_levels, chi_squared
  - `benches/icc.rs` (new) — srgb_v2 generation, ICC parse, ICC apply, ToneCurve gamma/table
  - `benches/spectral.rs` (new) — spd_to_xyz, xyz_to_cct, wavelength_to_xyz, blackbody_spd, color_temperature_to_rgb, cie_cmf_at, CRI, XYZ roundtrip
  - Expanded `benches/blend.rs` — all 12 blend modes group, ARGB pixel, ARGB row
  - Expanded `benches/color_convert.rs` — BT.2020 encode/decode, argb_to_nv12, rgba8_to_argb8, rgba8_to_rgb8, rgba8_to_rgbaf32
  - Expanded `benches/transform.rs` — bicubic resize, perspective transform
  - Expanded `benches/composite.rs` — unpremultiply, apply_mask, gradient_radial, gradient_angled, composite_at_argb

### Changed

- Added `spectral` feature flag (depends on prakash 0.23.3, spectral feature only)
- `full` feature now includes `spectral`
- **GPU batched dispatch** — `GpuChain` builder for chaining multiple GPU operations without CPU readback between steps (invert, grayscale, brightness_contrast, saturation, gaussian_blur, blend, noise, dissolve, fade, wipe, crop, resize, flip)
- **GPU noise generation** — `gpu_noise_gaussian` compute shader using PCG hash + Box-Muller transform for deterministic Gaussian noise
- **GPU transition shaders** — `gpu_dissolve`, `gpu_fade`, `gpu_wipe` compute shaders for cross-dissolve, fade-to-black, and horizontal wipe transitions
- **GPU geometry shaders** — `gpu_crop`, `gpu_resize` (nearest + bilinear), `gpu_flip_horizontal`, `gpu_flip_vertical` compute shaders
- **Visual regression tests** — 10 deterministic pixel-level regression tests: gradient blur smoothness, checkerboard resize, invert idempotency, premultiply roundtrip precision, Gaussian blur symmetry, HSL hue shift 360 identity, color balance neutral, crop+resize composition, Screen blend commutativity, YUV roundtrip color fidelity
- **Extended fuzz campaigns** — 5 new fuzz targets (blur, LUT, ICC, composite, transform) added to existing 3 (blend, convert, filter) for 8 total
- **Consistent error messages** — standardized all `InvalidFormat` errors to `"<operation>: expected <format>, got <actual>"` across filter, composite, histogram, convert, transform, and GPU modules
- Bumped `ai-hwaccel` from 0.21.3 to 0.23.3 — `HwReport` now exposes `gpu_free_memory_mb`, `gpu_utilization_percent`, `temperature_c`; `should_use_gpu()` checks free VRAM and GPU utilization before recommending offload
- Added field-level doc comments to all public struct fields (`LinRgba`, `Srgba`, `CieXyz`, `Cmyk`, `PixelBuffer`, `Affine`) — 100% public API documented
- Version aligned with prakash ecosystem at 0.23.3
- Roadmap fully cleared — all backlog items completed

## [0.21.4] — 2026-03-21

### Changed

- Bumped `ai-hwaccel` dependency from 0.20 to 0.21.3
- Cleaned up roadmap: removed completed items, added ai-hwaccel 0.21.3 review task

## [0.21.3] — 2026-03-21

### Fixed

- **Compositing precision** — replaced `>> 8` (divide-by-256) with proper `div255` rounding across all blend and composite paths (scalar, SSE2, AVX2, NEON), eliminating ~0.4% cumulative brightness loss per compositing pass
- **NEON brightness OOB read** — `simd_pixels` now rounds to multiple of 8 (matching `vld4_u8` stride), preventing buffer overread on aarch64
- **ARGB fast-path alpha** — `composite_at` and `composite_at_argb` fast-path now requires full opacity, preventing raw source alpha from bypassing opacity adjustment
- **ICC LUT index ordering** — CLUT indexing corrected to have B channel varying fastest per ICC spec, fixing color output for real-world LUT-based profiles
- **BT.709 Y coefficients** — changed from (54, 183, 18) sum=255 to (54, 183, 19) sum=256 so white correctly maps to Y=255
- **YUV420p odd-dimension buffer sizing** — `buffer_size()` and all conversion functions now use `div_ceil(2)` for chroma plane dimensions, fixing undersized buffers for odd-width/height images
- **Histogram `bins=0` panic** — `luminance_histogram()` now returns an error instead of panicking on zero bins
- **`auto_levels` color shift** — switched from luminance-based offset to per-channel min/max stretching, preventing color distortion
- **`auto_white_balance` extreme scale** — raised near-zero threshold from 0.5 to 5.0 and clamped scale factors to [0.5, 3.0]
- **GPU `pixel_count` truncation** — GPU pipeline functions now return an error instead of silently truncating images exceeding `u32::MAX` pixels
- **ICC tag offset overflow** — `parse_tag_table` now validates offset+size against profile length, preventing potential bounds bypass on crafted profiles
- **Gradient interpolation clamp** — `gradient_linear` and `gradient_linear_angled` now clamp interpolated values before u8 cast
- Added `debug_assert!` guards on NEON Y-plane coefficient values to catch u8 truncation
- Updated `srgb_v2_profile` docs to clarify gamma 2.2 is a v2 approximation of the piecewise sRGB TRC

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
