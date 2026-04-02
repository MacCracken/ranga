# Roadmap — Audit Backlog

Items identified during P(-1) scaffold hardening audit (2026-04-02).

## Completed

- [x] **pixel.rs** — `PixelBuffer` fields `pub(crate)`, public accessors added
- [x] **pixel.rs** — `rows()`/`rows_mut()` debug_assert for planar formats
- [x] **pixel.rs** — `BufferPool::acquire` best-fit instead of first-fit
- [x] **pixel.rs** — `checked_buffer_size()` for overflow-safe dimension validation
- [x] **pixel.rs** — `#[inline]` on all accessors, `#[must_use]` on `set_rgba`
- [x] **composite.rs** — `div255` consistency in `premultiply_alpha`/`apply_mask`
- [x] **histogram.rs** — `chi_squared` length validation (returns `Result`)
- [x] **filter.rs** — Bilateral sigma>0 validation, spatial weight precomputation
- [x] **filter.rs** — Levels gamma>0 validation
- [x] **filter.rs** — Flood fill rewritten with scanline algorithm
- [x] **filter.rs** — Median filter rewritten with histogram-based (Huang) approach
- [x] **filter.rs** — Vertical blur parallelized
- [x] **blend.rs** — Doc fix: straight alpha, not premultiplied
- [x] **blend.rs** — SIMD slice validation: `debug_assert` → runtime check
- [x] **color.rs** — `#[inline]` on all `From` impls and free functions
- [x] **color.rs** — `color_temperature` NaN guard
- [x] **color.rs** — Alpha loss documented on XYZ/Oklab conversions
- [x] **icc.rs** — `tag_count` capped at 1024, `grid_size` capped at 64
- [x] **icc.rs** — `read_*` helpers return `Result` (no more panics)
- [x] **icc.rs** — `#[inline]` on `ToneCurve::apply` and read helpers
- [x] **transform.rs** — Perspective NaN propagation fixed
- [x] **transform.rs** — Resize 0-dimension source guarded
- [x] **convert.rs + filter.rs** — Fake SSE2 removed, scalar fallback
- [x] **gpu/** — Migrated from raw wgpu to mabda (pipeline cache, shader cache, buffer helpers)
- [x] **gpu/** — All `expect()`/`unwrap()` replaced with `Result` propagation
- [x] **gpu/buffer.rs** — `GpuBuffer` stores format (no more hardcoded Rgba8)
- [x] **gpu/shaders.rs** — `pack_rgba` uses `round()` instead of floor+0.5 bias
- [x] **gpu/shaders.rs** — Luminance coefficients standardized to BT.709
- [x] **gpu/shaders.rs** — Noise R/B channels decorrelated (independent Box-Muller pair)
- [x] **Cargo.toml** — License AGPL→GPL, deps updated, deny.toml/vet cleaned

## Completed — SIMD

- [x] **convert.rs** — Real SSE2 Y-row via `_mm_madd_epi16`
- [x] **filter.rs** — Real SSE2 grayscale via `_mm_madd_epi16`
- [x] **blend.rs** — `blend_row` API for all modes (Normal dispatches to SIMD)
- [x] **blend.rs** — SSE2 path for `blend_row_normal_argb`

## Remaining — Architecture

- [ ] **icc.rs** — Add ICC `para` curve types 1, 2, 4 (only 0 and 3 supported)
- [ ] **transform.rs** — `perspective_transform` add `ScaleFilter` parameter
- [ ] **convert.rs** — SIMD for YUV-to-RGBA inverse conversions
