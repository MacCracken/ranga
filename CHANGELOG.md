# Changelog

## [Unreleased]

## [2.0.0] — 2026-08-19

**The Cyrius port.** ranga is now written in Cyrius. The Rust line ends at
1.0.1 and is preserved at `rust-old/` as the parity oracle. Major bumps without
resetting the series, matching every prior AGNOS port (vidya 1.x→2.0.0,
prakash 1.2.0→2.0.0, mabda 1.0.0→2.0.0, ai-hwaccel 1.2.0→2.0.0).

**This is not a source-compatible upgrade.** Cyrius has no methods, no traits and
no `Drop`, so every `X::y()` is now a free `x_y()`, every owning type has an
explicit `*_free`, and errors are `i64` codes rather than a `RangaError` enum.
Consumers rewrite against the new surface; `docs/development/parity-rust-v-cyrius.md`
maps it item by item.

### Added

- **`spectral` and `hwaccel` as optional features**, as they were in Rust
  (`default = []`). `spectral` bridges **prakash 2.2.3**; `hwaccel` bridges
  **ai-hwaccel 2.3.17**.
- **`gpu` as an optional feature over mabda 4.0.9**, with native AMD (amdgpu /
  GFX9) and NVIDIA (nouveau / SM75) backends and wgpu as the fallback.
- **A SPIR-V emitter** (`src/gpu_spirv.cyr`). mabda's native backends accept
  SPIR-V and nothing else, and nothing in the stack translates WGSL, so ranga
  emits SPIR-V directly. Eleven of the fourteen GPU operations run natively;
  all fourteen have WGSL for the wgpu path.
- **`ranga_error_message`** — the `RangaError` Display strings, which the port
  had been missing entirely.
- **`bytevec.cyr`** — a byte-granular `Vec<u8>` substitute. Cyrius's `lib/vec.cyr`
  stores i64 slots, so a 4-byte pixel would occupy 32.
- Four bundle profiles: `dist/ranga.cyr` (core, no external deps) plus
  `-spectral`, `-hwaccel` and `-gpu`.

### Changed

- **`ColorSpace` gains `Bt601` and `Bt709`** and renumbers to match Rust
  exactly. The port had dropped both and let its own `CieLab`/`Oklab` additions
  take values 3-6, so a `ColorSpace` persisted by 1.0.1 read back as a different
  space. The first seven discriminants are now Rust's; ranga's additions are
  numbered above that range.
- **`PixelView`/`PixelViewMut` require an EXACT buffer length**, as Rust did.
  The port had accepted any buffer large enough, which silently widened the
  contract.
- **GPU grayscale is BT.709 where CPU grayscale is BT.601** — inherited from
  Rust, which had the same split. Documented rather than reconciled.

### Not ported

- **`gpu_blend`, `gpu_noise_gaussian`, `gpu_gaussian_blur` and bilinear
  `gpu_resize`** report `RG_GPU_ERR_UNSUPPORTED` on the native backends and run
  on wgpu. Blocked by limits in mabda's SPIR-V→GFX9 lowering — register
  pressure for the first three, `OpLoopMerge` rejection for blur. Filed at
  `mabda/docs/development/issues/2026-08-19-native-spirv-compile-limits.md`.
- **`GpuBuffer::download_async`** — existed to work around a Rust borrow-checker
  constraint; the Cyrius mabda exposes no async readback to build on.
- **`Lut3d::from_cube`** — the `.cube` file parser. The LUT itself is ported;
  loading one from disk is not.
- **`PixelBuffer::into_data`**, **`GpuContext::new_with_hwaccel`**, and the
  `FromStr` inverses of `blend_mode_name` / `color_space_name` / `pixel_format_name`.
- **Serde codecs for the four enums** are hand-written. `#derive(Serialize)`
  works on a Cyrius struct but silently emits nothing on an enum; filed at
  `cyrius/docs/development/issues/2026-08-19-derive-serialize-on-enum-emits-nothing.md`.

### Toolchain

- **Cyrius pin 6.5.29** (was 6.5.27). Two defects this port filed were fixed in
  between: decimal float literals past ~9 significant digits parsing to a
  silently wrong value (6.5.28), and `distlib` profile `.deps` sidecars written
  empty (6.5.29). ranga keeps its hex bit patterns for the Oklab matrices —
  they are correct and pinned by the white-point assertions, and rewriting
  eighteen verified constants would be churn with a real chance of introducing
  the very error class the hex was adopted to avoid.
- **`#derive(Serialize)` on an enum still emits nothing**, re-tested at the
  6.5.29 pin. The four enum codecs stay hand-written.

### Fixed — found by the final parity sweep

- **`oklab(black)` returned `(NaN, NaN, NaN)`.** `ganita_f32_cbrt(0.0)` is NaN
  where Rust's `f32::cbrt(0.0)` is `0.0`, and it propagated through the whole
  Oklab M2 matrix. Rust pins this in a test the port had not ported.
- **`apply_lut3d` truncated where Rust rounds half-up**, so half of all grey
  levels came out one count low.
- **An ICC LUT profile with `grid_size == 0` read below its allocation.** Now
  rejected — a deliberate divergence from Rust, which underflows there.

Known remaining divergences — chiefly f32-vs-f64 width in the scalar tails, and
NaN handling where the port is NaN-correct and Rust's casts saturate — are
enumerated in `docs/development/parity-rust-v-cyrius.md`.

### Testing

**1,878 assertions across 19 suites, 0 lint warnings.** Expected values come
from Python oracles replicating the exact Rust formula, not from reading the
Cyrius back. Every module is mutation-tested once it goes green.

The GPU path is hardware-verified on an AMD Cezanne (gfx90c): SPIR-V emitted by
ranga, lowered to GFX9 by mabda, dispatched, and read back byte-exact against
f32 oracles.

### Upstream issues filed during the port

Two resolved in cycc during the port — decimal float literals past ~9
significant digits parsing to a silently wrong value, and `fl_calloc` re-zeroing
mmap'd pages a byte at a time. Three remain open: mabda's native SPIR-V limits,
`#derive(Serialize)` on enums, and `distlib` profile sidecars written empty.


## [1.0.1] — 2026-08-13

**Final Rust release.** ranga continues in Cyrius alongside the rest of the AGNOS
stack (mabda, prakash, ai-hwaccel have already ported). This release brings the
Rust line current on toolchain and dependencies and closes out the crates.io
publishing pipeline. No API changes — 1.0.0 consumers upgrade in place.

### Changed

- **Toolchain**: verified against Rust 1.97.1 (was 1.96.0). MSRV stays **1.89**, edition stays 2024 — the floor is deliberately unchanged so consumers still on the Rust line can take this release during the port.
- **Release pipeline**: `cargo publish` removed from `.github/workflows/release.yml`. The `publish` job is replaced by `verify`, which keeps the VERSION/Cargo.toml/tag consistency gate; `release` now depends on `[ci, build, verify]`. Releases are GitHub artifacts only.
- **Dependencies** — direct:
  - `ai-hwaccel` 1.0.0 → 1.2.0
  - `criterion` 0.5.1 → 0.8.2 (dev)
  - `rayon` 1.11.0 → 1.12.0
  - `serde` 1.0.228 → 1.0.229, `serde_json` 1.0.149 → 1.0.151
  - `thiserror` 2.0.18 → 2.0.20
  - `wgpu` 29.0.1 → 29.0.4 (patch only — see below)
- **Dependencies** — transitive: `naga`/`wgpu-core`/`wgpu-hal`/`wgpu-types` 29.0.1 → 29.0.4, `bytemuck` 1.25.0 → 1.25.2, `getrandom` 0.4.2 → 0.4.3, plus 60+ others. Dependency graph shrank from 210 to 196 crates.
- **`hwaccel::probe`**: migrated to the ai-hwaccel 1.2 iterator API — `AcceleratorRegistry::available()` and `by_family()` now return `impl Iterator` rather than slices, so the `.iter()` hops are gone. Both already filter on `available`, so probe results are unchanged.
- **Benchmarks**: `black_box` imported from `std::hint` instead of `criterion::black_box`, which criterion deprecated in 0.6. Applies to all 10 bench suites.

### Added

- `median_r5_512x512` benchmark. The 1.0.0 Huang rewrite made `median` O(n·r) per channel instead of O(n·r²·log r²), but only r=1 was ever measured — the one radius where a 256-bin-per-channel histogram costs more to set up than sorting the 9-sample window outright. Measuring a second radius keeps the tradeoff visible in the record rather than leaving a bare regression.

### Fixed

- **RUSTSEC-2026-0204** (vulnerability): invalid pointer dereference in `crossbeam-epoch`'s `fmt::Pointer` impl, reached through `rayon` → `crossbeam-deque`. Resolved by 0.9.18 → 0.9.20. This was the one hard `cargo audit`/`cargo deny` failure on 1.0.0's lockfile.
- **RUSTSEC-2026-0097** (unsoundness): `rand` unsound with a custom logger using `rand::rng()`. Resolved by 0.9.2 → 0.9.5.
- **RUSTSEC-2026-0190** (unsoundness): `anyhow` `Error::downcast_mut()`. `anyhow` dropped from the graph entirely by the criterion 0.8 upgrade.
- **Benchmarks measuring nothing**: `hsl_roundtrip`, `xyz_conversion_roundtrip`, and `pixel_view_create` bound their results to `_` instead of returning them from the `iter` closure, so nothing black-boxed the output and the optimizer was free to delete the work being timed. `hsl_roundtrip` was recording **257 picoseconds** — under one clock cycle — against a real cost of 19.6ns. The bug predates this release but was masked by criterion 0.5's weaker `black_box`; `std::hint::black_box` let the optimizer see through it completely. All three now return their values.
- **Docs**: README listed `ai-hwaccel 0.23.3` in the feature and capability tables (actual: 1.2); the GPU compute guide's install snippet still said `version = "0.20"`. The median benchmark comment still described the pre-1.0.0 O(n·radius²) complexity.

### Performance

Neutral. Full 134-benchmark sweep recorded to `benches/history.csv`. The last
recorded sweep was `04f74df` (2026-04-02), eight commits back, so a naive diff
against it attributes that intervening work to this release — it shows 50
benchmarks >10% faster (the SIMD conversion and blur paths) and 21 >10% slower.
Neither belongs to 1.0.1.

An A/B against HEAD-without-these-changes, same machine and session, confirms the
dependency and toolchain moves are performance-neutral:

| benchmark | Apr baseline | HEAD today | 1.0.1 today |
|---|---|---|---|
| `median_r1_512x512` | 15.347 ms | 30.549 ms | 29.341 ms |
| `affine_rotate_512x512` | 5.9141 ms | 7.2336 ms | 7.2699 ms |
| `resize_bicubic_1080p_to_720p` | 91.485 ms | 105.37 ms | 104.83 ms |
| `cmyk_roundtrip` | 10.155 ns | 13.417 ns | 13.076 ns |

Every apparent regression reproduces at HEAD without this release's changes. The
`median_r1` movement is the 1.0.0 Huang rewrite's small-radius tradeoff, which
landed in `2aa3b82` after the last recorded sweep; `median_r5_512x512` (40.317 ms)
is added to show the radius scaling the rewrite was for. The remainder tracks the
other post-baseline commits and machine conditions — this run was taken under a
`powersave` governor at load ~4.5, so absolute figures are not comparable to the
April run's unknown conditions. Sweeps for the port should be taken on a quiesced
machine.

### Pinned

- `wgpu` stays on **29** and `pollster` on **0.4**. mabda 1.0.0 — the last Rust release of the AGNOS GPU foundation — links `wgpu ^29` and `pollster ^0.4`. Moving ranga to wgpu 30 / pollster 1.0 would resolve a second copy of wgpu into the tree and `mabda::GpuContext`'s types would no longer unify with ours. Since mabda's Rust line is frozen, these majors are frozen with it. Rationale recorded inline in `Cargo.toml`.

## [1.0.0] — 2026-04-02

P(-1) scaffold hardening pass — full audit and fix cycle across all modules.

### Changed

- **License**: AGPL-3.0-only → GPL-3.0-only (Cargo.toml, fuzz/Cargo.toml, CONTRIBUTING.md, README.md)
- **GPU backend**: migrated from raw wgpu to mabda 1.0 (pipeline cache, shader cache, buffer helpers, pollster)
  - `GpuContext` wraps `mabda::GpuContext` — no more `RefCell`, raw pointers, or custom `block_on`
  - `GpuBuffer` uses `mabda::buffer` helpers; now stores `PixelFormat` (was hardcoded Rgba8)
  - All `gpu_*` functions and `GpuChain` take `&mut GpuContext` (was `&GpuContext`)
  - All `expect()`/`unwrap()` in GPU code replaced with `Result` propagation
- **PixelBuffer**: fields now `pub(crate)` — use `data()`, `data_mut()`, `into_data()`, `width()`, `height()`, `format()` accessors
- **BufferPool::acquire**: best-fit allocation (was first-fit)
- **Flood fill**: rewritten with scanline algorithm (was naive 4-neighbor push with O(4n) stack bloat)
- **Median filter**: rewritten with histogram-based Huang approach — O(n*r) per channel (was O(n*r^2*log(r^2)))
- **Bilateral filter**: spatial Gaussian weights precomputed into table (was `exp()` per sample per pixel)
- **Vertical blur**: parallelized with rayon (horizontal was already parallel)
- **SSE2**: replaced fake SIMD stubs (extracted lanes → scalar) with real implementations
  - `compute_y_row` — `_mm_madd_epi16` horizontal dot-product, 2 pixels/iter
  - `grayscale` — `_mm_madd_epi16` luminance, 2 pixels/iter
  - `blend_row_normal_argb` — full SSE2 Porter-Duff, 2 pixels/iter
  - YUV-to-RGBA inverse (BT.601, BT.709, BT.2020, NV12) — 8 pixels/iter with U/V broadcast
- **GPU shaders**: `pack_rgba` uses `round()` (was `floor(x+0.5)`); saturation luminance standardized to BT.709; noise R/B decorrelated with independent Box-Muller pair
- **`chi_squared`**: returns `Result<f64, RangaError>` with length validation (was silent truncation)
- **`composite`**: `premultiply_alpha`/`apply_mask` use `div255` (was `/255`, off-by-one)
- **`blend`**: doc corrected to "straight alpha" (was "premultiplied"); SIMD slice checks upgraded from `debug_assert` to runtime guard
- **`perspective_transform`**: accepts `ScaleFilter` parameter (Nearest/Bilinear/Bicubic)
- **prakash**: 1.1.1 → 1.2 (version spec widened)
- **deny.toml**: removed stale advisory ignore, license entries, fuzz exception; cleaned cargo-vet imports

### Added

- `PixelFormat::checked_buffer_size()` — overflow-safe dimension validation
- `blend_row()` — row-level blend for any `BlendMode` (dispatches Normal to SIMD)
- ICC `para` curve types 1, 2, 4 (was only 0 and 3) — all 5 ICC parametric curve types now supported
- `#[inline]` on all `From` impls in `color.rs`, all `PixelBuffer`/`PixelView`/`PixelViewMut` accessors, `ToneCurve::apply`, ICC read helpers, filter scalars, spectral wrappers, GPU buffer accessors
- `#[must_use]` on `PixelBuffer::set_rgba`
- `scripts/bench-history.sh` — benchmark CSV tracking with Criterion output parsing
- `benches/history.csv` — baseline + post-hardening benchmark data

### Fixed

- **buffer_size overflow**: `checked_buffer_size()` prevents silent wrap on large dimensions; `buffer_size()` panics instead of wrapping
- **Perspective NaN**: degenerate projections (w≈0) now correctly skip pixels instead of indexing with NaN
- **Resize 0-dim source**: early return for `width==0 || height==0` prevents `usize` underflow
- **ICC parser**: `tag_count` capped at 1024, `grid_size` capped at 64 (prevents OOM from malformed profiles); `read_u32_be`/`read_u16_be`/`read_i32_be`/`read_s15fixed16` return `Result` (was panicking on OOB)
- **`rows()`/`rows_mut()`**: debug_assert rejects planar formats (Yuv420p, Nv12) instead of silently producing wrong slices
- **Bilateral filter**: `sigma=0` now returns error (was `-inf` coefficient)
- **Levels**: `gamma=0` now returns error (was `powf(infinity)`)
- **`color_temperature`**: NaN input returns neutral `[1.0, 1.0, 1.0]` (was propagating NaN through clamp)
- **GPU doctest**: `wgpu::Maintain::Wait` → `wgpu::PollType::wait_indefinitely()`

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
