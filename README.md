# Ranga

> रंग (Sanskrit: color, hue) — Core image processing library for Rust

Ranga provides shared image processing primitives for the AGNOS creative suite, eliminating duplicate implementations across [rasa](https://github.com/agnostos/rasa) (image editor), [tazama](https://github.com/MacCracken/tazama) (video editor), and [aethersafta](https://github.com/MacCracken/aethersafta) (compositor).

## Features

- **Color spaces** — sRGB, linear RGB, HSL with proper gamma conversion
- **Pixel buffers** — Unified `PixelBuffer` type with 6 format variants (RGBA8, ARGB8, RGB8, YUV420p, NV12, RgbaF32)
- **Blend modes** — 12 Porter-Duff blend modes (Normal, Multiply, Screen, Overlay, etc.)
- **Color conversion** — BT.601 fixed-point RGBA↔YUV420p, ARGB→NV12
- **Filters** — Brightness, contrast, saturation, levels, curves, grayscale, invert
- **Histograms** — Luminance and per-channel RGB histograms, chi-squared distance

## Usage

```rust
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::filter;
use ranga::convert;

// Create an RGBA buffer
let mut buf = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);

// Apply filters
filter::brightness(&mut buf, 0.1)?;
filter::contrast(&mut buf, 1.2)?;
filter::saturation(&mut buf, 1.1)?;

// Convert to YUV for video encoding
let yuv = convert::rgba_to_yuv420p(&buf)?;
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `simd` | Yes | SSE2/AVX2/NEON acceleration for blend and convert |
| `gpu` | No | wgpu-based GPU compute pipelines |

## License

AGPL-3.0-only
