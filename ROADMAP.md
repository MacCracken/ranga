# Roadmap — Audit Backlog

Items identified during P(-1) scaffold hardening audit (2026-04-02).
Ordered by priority within each category.

## MEDIUM — Correctness & Safety

- [ ] **pixel.rs:94-103** — Make `PixelBuffer` fields private, add accessors. Public fields allow invariant violation (data/dimensions mismatch). Biggest structural change — ripples through entire codebase.
- [ ] **pixel.rs:196-203** — `rows()`/`rows_mut()` silently produce wrong results for planar formats (Yuv420p, Nv12). Return `Result` or add debug assertions.
- [ ] **composite.rs:60-63,124** — `premultiply_alpha` and `apply_mask` use `/255` instead of `div255`, inconsistent with compositing pipeline (off-by-one for many inputs).
- [ ] **histogram.rs:116** — `chi_squared` silently truncates mismatched-length inputs. Should return error.
- [ ] **icc.rs:410** — `grid_size.pow(3)` from untrusted input can cause large allocation (~100MB). Add upper bound.
- [ ] **filter.rs:1183-1184** — Bilateral filter `sigma=0` causes `-inf` coefficient, silent degradation. Validate sigma > 0.
- [ ] **filter.rs:249** — Levels `gamma=0` causes `powf(infinity)`. Validate gamma > 0.
- [ ] **blend.rs:27** — Doc comment says "premultiplied alpha" but `blend_pixel` operates on straight alpha. Fix docs.
- [ ] **blend.rs:429-430** — SIMD slice validation uses `debug_assert` only — could OOB in release with bad input. Add runtime check.
- [ ] **gpu/buffer.rs:127,200** — Hardcoded `PixelFormat::Rgba8` on download — `GpuBuffer` should store format.
- [ ] **gpu/shaders.rs:22-23** — `pack_rgba` rounding bias (`floor(x+0.5)`) may differ from CPU rounding.
- [ ] **gpu/context.rs:198-262** — Raw pointer from `RefCell` borrow relies on HashMap insertion stability. Replace with stable-index collection.

## MEDIUM — Performance

- [ ] **pixel.rs:528** — `BufferPool::acquire` finds first-fit, not best-fit. Use `min_by_key` on capacity.
- [ ] **filter.rs:1394-1424** — Flood fill pushes duplicates causing O(4n) memory bloat. Check visited before push, or use scanline fill.
- [ ] **filter.rs:540-569** — Vertical blur pass not parallelized (horizontal is).
- [ ] **filter.rs:1127-1145** — Median filter is O(n*r^2*log(r^2)) per pixel. Histogram-based approach (Huang et al.) would be O(n*r).
- [ ] **filter.rs:1187-1231** — Bilateral filter spatial Gaussian weights recomputed per pixel. Precompute into table.
- [ ] **convert.rs:all YUV-to-RGBA** — All inverse conversions are scalar per-pixel, no SIMD.
- [ ] **blend.rs:102** — Non-Normal blend modes all scalar f32 per-pixel, no SIMD row variants.
- [ ] **gpu/pipeline.rs:build_shader** — Shader string allocated on every dispatch even on cache hits. Move inside cache-miss path.

## LOW — Annotations & Docs

- [ ] **color.rs:many** — Missing `#[inline]` on all `From` impls — blocks cross-crate inlining.
- [ ] **pixel.rs:176,242,269,61** — Missing `#[inline]` on `pixel_count`, `get_rgba`, `set_rgba`, `buffer_size`.
- [ ] **icc.rs:95,220,311,365** — Missing `#[inline]` on `ToneCurve::apply`, `IccProfile::apply`, `IccLutProfile::apply`, `apply_curve`.
- [ ] **filter.rs:60,340,475,998** — Missing `#[inline]` on `brightness_scalar`, `grayscale_scalar`, `build_gaussian_kernel`, `Xorshift64` methods.
- [ ] **transform.rs:191** — Missing `#[inline]` on `validate_rgba8`.
- [ ] **spectral.rs:54,62,73,80,85,94** — Missing `#[inline]` on all wrapper functions.
- [ ] **gpu/buffer.rs:210-231** — Missing `#[inline]` on trivial accessors.
- [ ] **pixel.rs:269** — Missing `#[must_use]` on `set_rgba` return value.
- [ ] **histogram.rs:146,224** — Missing `#[must_use]` on `equalize`/`auto_levels` Results.
- [ ] **color.rs:479,741** — Alpha silently lost in XYZ/Oklab conversions. Document behavior.
- [ ] **color.rs:843** — `color_temperature` doesn't guard against NaN input (clamp passes NaN).
- [ ] **gpu/shaders.rs:152-156 vs 213-219** — Inconsistent luminance coefficients: grayscale uses BT.709, saturation uses BT.601.
- [ ] **gpu/shaders.rs:598-601** — Noise R/B channel correlation. Generate third random value for B.

## LOW — Architecture

- [ ] **icc.rs:806** — Only ICC `para` curve types 0 and 3 supported. Add types 1, 2, 4.
- [ ] **transform.rs:696** — `perspective_transform` hardcodes bilinear. Add `ScaleFilter` parameter.
- [ ] **gpu/context.rs:83** — `RefCell<PipelineCache>` prevents `GpuContext` from being `Sync`. Consider `Mutex` for multi-threaded GPU dispatch.
- [ ] **gpu/context.rs:416-435** — Spin-loop `block_on` with no timeout. Add timeout mechanism.

## Real SSE2 SIMD (deferred)

- [ ] **convert.rs** — Write proper SSE2 Y-row computation using `_mm_madd_epi16` for horizontal dot-product.
- [ ] **filter.rs** — Write proper SSE2 grayscale using channel deinterleave + vectorized multiply-accumulate.
- [ ] **blend.rs** — Add SIMD row-blend variants for non-Normal blend modes.
- [ ] **blend.rs:497** — Add SIMD path for `blend_row_normal_argb`.
