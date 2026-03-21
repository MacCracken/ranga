# Architecture Overview

Ranga is a pure-Rust image processing library providing shared primitives for
the AGNOS creative suite. It replaces duplicate color math, blending,
conversion, and filter code across rasa (image editor), tazama (video editor),
and aethersafta (compositor).

## Module Map

```
ranga
├── color        — Color spaces (sRGB, linear, HSL, XYZ, Lab, P3, CMYK)
│                  Delta-E distance, color temperature
├── pixel        — PixelBuffer, PixelView, PixelFormat, BufferPool
├── blend        — 12 Porter-Duff blend modes (SSE2/AVX2/NEON SIMD)
├── convert      — Pixel format conversion (BT.601, BT.709, NV12, RGB↔RGBA, etc.)
├── filter       — 23 image filters (blur, sharpen, color, noise, LUT, etc.)
├── transform    — Geometry: crop, resize, affine, flip
├── composite    — Layer compositing: masks, transitions, fill, positioned blend
├── histogram    — Luminance/RGB histograms, chi-squared distance
├── icc          — ICC profile parsing (matrix-based v2/v4)
├── gpu          — wgpu compute shaders (blend, filters, blur) [feature: gpu]
└── hwaccel      — GPU detection via ai-hwaccel [feature: hwaccel]
```

## Data Flow

All operations center on `PixelBuffer` — a validated `Vec<u8>` with known
format and dimensions. The typical pipeline is:

```
Input bytes → PixelBuffer::new() → [filter/blend/convert] → output bytes
```

For zero-copy integration with downstream projects:

```
Existing &[u8] → PixelView::new() → read-only operations
Existing &mut [u8] → PixelViewMut::new() → in-place operations
```

For video frame pipelines, `BufferPool` avoids per-frame allocation:

```
pool.acquire(size) → process → pool.release(buf) → reuse next frame
```

## Pixel Format Convention

All filter, blend, and composite operations require RGBA8 (4 bytes/pixel,
straight alpha). Conversion functions handle format interchange:

- `rgb8_to_rgba8` / `rgba8_to_rgb8` — add/strip alpha
- `argb8_to_rgba8` / `rgba8_to_argb8` — channel reorder (aethersafta uses ARGB)
- `rgba_to_yuv420p` / `yuv420p_to_rgba` — video YUV (BT.601 or BT.709)
- `rgbaf32_to_rgba8` / `rgba8_to_rgbaf32` — HDR float conversion

## SIMD Strategy

SIMD acceleration is gated behind the `simd` feature (default on). The blend
hot path dispatches at runtime:

1. Check `is_x86_feature_detected!("avx2")` → AVX2 path (4 pixels/iter)
2. Else SSE2 baseline on x86_64 (2 pixels/iter)
3. NEON baseline on aarch64 (8 pixels/iter)
4. Scalar fallback on other targets

All SIMD paths produce results within +/-1 of the scalar path (verified by
equivalence tests).

## GPU Strategy

The `gpu` feature provides wgpu compute shaders for large-image operations.
The `GpuContext` caches compiled pipelines to amortize shader compilation.

**CPU is faster below ~1080p** due to upload/download overhead. Use
`hwaccel::should_use_gpu(w, h)` for automatic selection.

GPU shaders operate on packed u32 RGBA (not f32) to avoid CPU-side format
conversion. Each shader includes pack/unpack helpers.

## Feature Flags

| Flag | Deps added | What it enables |
|------|-----------|----------------|
| `simd` (default) | none | SSE2/AVX2/NEON blend acceleration |
| `gpu` | wgpu | Compute shaders for blend, filter, blur |
| `hwaccel` | ai-hwaccel | GPU detection and capability query |
| `parallel` | rayon | Row-parallel blur |
| `full` | all above | Everything |

## Error Handling

All fallible operations return `Result<_, RangaError>`. The error enum is
`#[non_exhaustive]` for forward compatibility. GPU errors convert via
`From<GpuError> for RangaError`.

## Design Principles

1. **No I/O** — ranga processes pixels, never touches files or network
2. **No C FFI** — pure Rust + `std::arch` SIMD (see ADR-001)
3. **Format validation** — every operation checks `PixelFormat` before processing
4. **Odd-dimension safe** — YUV/NV12 chroma subsampling handles non-even sizes
5. **Deterministic** — noise functions accept seeds for reproducible output
6. **Zero-copy when possible** — `PixelView` borrows existing buffers
