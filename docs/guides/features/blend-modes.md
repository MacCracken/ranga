# Blend Modes Guide

## Supported Modes

| Mode | Formula | Effect |
|------|---------|--------|
| Normal | `src` | Direct replacement |
| Multiply | `src × dst` | Darken (product) |
| Screen | `1 - (1-src)(1-dst)` | Lighten (inverse multiply) |
| Overlay | Multiply/Screen hybrid | Contrast boost |
| Darken | `min(src, dst)` | Keep darker |
| Lighten | `max(src, dst)` | Keep lighter |
| ColorDodge | `dst / (1-src)` | Brighten highlights |
| ColorBurn | `1 - (1-dst)/src` | Darken shadows |
| SoftLight | W3C formula | Gentle contrast |
| HardLight | Overlay with swapped inputs | Strong contrast |
| Difference | `|src - dst|` | Invert-like |
| Exclusion | `src + dst - 2×src×dst` | Softer difference |

## Single Pixel Blend

```rust
use ranga::blend::{BlendMode, blend_pixel};

let result = blend_pixel(
    [255, 0, 0, 200],   // source RGBA
    [0, 0, 255, 255],   // destination RGBA
    BlendMode::Normal,
    255,                  // opacity (0–255)
);
```

Porter-Duff "source over" compositing with per-pixel alpha and per-layer opacity.

## Row-Level Blend (SIMD)

```rust
use ranga::blend::blend_row_normal;

let src = vec![128u8; 1920 * 4]; // 1920 RGBA pixels
let mut dst = vec![64u8; 1920 * 4];
blend_row_normal(&src, &mut dst, 200); // opacity 200/255
```

Automatically dispatches to SSE2, AVX2, or NEON depending on platform. ~2x
faster than scalar on x86_64.

## GPU Blend

```rust,no_run
use ranga::gpu::{GpuContext, gpu_blend};
use ranga::blend::BlendMode;

let ctx = GpuContext::new()?;
gpu_blend(&ctx, &src_buf, &mut dst_buf, BlendMode::Multiply, 0.8)?;
```

All 12 modes in a single parameterized WGSL shader. Effective for 4K+ images.

## Positioned Composite

For layer-based compositing (aethersafta pattern):

```rust
use ranga::composite;

composite::composite_at(&layer, &mut canvas, 100, 50, 0.8)?;
```

Handles clipping, per-pixel alpha, and per-layer opacity. Supports negative
positions (partial visibility).
