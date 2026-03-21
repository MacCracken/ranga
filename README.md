# Ranga

> रंग (Sanskrit: color, hue) — Core image processing library for Rust

Ranga provides shared image processing primitives for the [AGNOS](https://github.com/MacCracken) creative suite, eliminating duplicate color math, blending, conversion, and filter implementations across [rasa](https://github.com/agnostos/rasa) (image editor), [tazama](https://github.com/MacCracken/tazama) (video editor), and [aethersafta](https://github.com/MacCracken/aethersafta) (compositor).

**Pure Rust core** — no C FFI. SIMD acceleration via `std::arch`, GPU compute via wgpu.

## Features

| Area | Capabilities |
|------|-------------|
| **Color spaces** | sRGB, linear RGB, HSL, CIE XYZ, CIE L\*a\*b\*, Display P3, CMYK |
| **Pixel buffers** | `PixelBuffer` with 6 formats, zero-copy `PixelView`, `BufferPool` |
| **Blend modes** | 12 Porter-Duff modes with SSE2/AVX2/NEON SIMD (2x throughput) |
| **Conversion** | BT.601, BT.709, NV12, RGB8↔RGBA8, ARGB8↔RGBA8, RgbaF32↔RGBA8 |
| **Filters** | 17 filters: blur, sharpen, hue shift, color balance, 3D LUT, vignette, noise |
| **Color science** | Delta-E (CIE76/94/2000), color temperature, ICC profile parsing |
| **Histograms** | Luminance, per-channel RGB, chi-squared distance |
| **GPU compute** | wgpu shaders for blend, filters, blur with pipeline caching |
| **Hardware** | GPU detection via ai-hwaccel, automatic CPU/GPU crossover |

## Quick Start

```toml
[dependencies]
ranga = "0.20"
```

```rust
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::filter;
use ranga::convert;

// Create an RGBA buffer
let mut buf = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);

// Apply filters
filter::brightness(&mut buf, 0.1)?;
filter::contrast(&mut buf, 1.2)?;
filter::gaussian_blur(&buf, 3)?;

// Convert to YUV for video encoding
let yuv = convert::rgba_to_yuv420p(&buf)?;
```

### Color Science

```rust
use ranga::color::*;

// sRGB → CIE Lab
let lab: CieLab = Srgba { r: 128, g: 64, b: 200, a: 255 }.into();

// Delta-E color distance
let distance = delta_e_ciede2000(&lab, &CieLab { l: 50.0, a: 0.0, b: 0.0 });

// Display P3 → sRGB
let (r, g, b) = p3_to_linear_srgb(0.8, 0.3, 0.5);

// Color temperature
let [r, g, b] = color_temperature(3200.0); // warm light
```

### GPU Compute

```rust,no_run
use ranga::gpu::{GpuContext, gpu_blend, gpu_gaussian_blur};
use ranga::blend::BlendMode;

let ctx = GpuContext::new()?;
gpu_blend(&ctx, &src, &mut dst, BlendMode::Multiply, 0.8)?;
let blurred = gpu_gaussian_blur(&ctx, &buf, 5)?;
```

### Zero-Copy Integration

```rust
use ranga::pixel::{PixelView, PixelFormat};

// Borrow existing buffer from rasa/tazama/aethersafta — no copy
let view = PixelView::new(&existing_bytes, 1920, 1080, PixelFormat::Rgba8)?;
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `simd` | Yes | SSE2/AVX2/NEON acceleration for blend operations |
| `gpu` | No | wgpu compute shaders (Vulkan/Metal) |
| `hwaccel` | No | GPU detection via ai-hwaccel |
| `parallel` | No | Rayon row-parallel blur |
| `full` | No | All features |

## Performance

Benchmarks at 1080p (1920x1080) on the host machine:

| Operation | Time |
|-----------|------|
| Blend row (1920px, SIMD) | 2.9 µs |
| Blend row (1920px, scalar) | 5.7 µs |
| Invert | 1.0 ms |
| Grayscale | 2.2 ms |
| Gaussian blur (r=3) | ~50 ms |
| RGBA→YUV420p BT.601 | 2.1 ms |
| Color temperature | 5.3 ns |
| Delta-E CIEDE2000 | 109 ns |

Run benchmarks: `cargo bench` or `cargo bench --all-features`

## Documentation

### Guides
- [Architecture Overview](docs/architecture/overview.md) — module map, data flow, design principles
- [Performance Guide](docs/guides/performance.md) — CPU/GPU crossover, SIMD tiers, benchmarking
- [Testing Guide](docs/guides/testing.md) — test matrix, coverage, fuzzing, CI
- [Troubleshooting](docs/guides/troubleshooting.md) — common build and runtime issues
- [Migration Guide](docs/development/migration-guide.md) — adopting ranga in rasa/tazama/aethersafta

### Feature Guides
- [Color Science](docs/guides/features/color-science.md) — color spaces, Delta-E, ICC, temperature
- [Blend Modes](docs/guides/features/blend-modes.md) — 12 modes, SIMD, GPU, positioned composite
- [Filters](docs/guides/features/filters.md) — 23 filters with usage patterns and performance tips
- [Transforms](docs/guides/features/transforms.md) — crop, resize, affine, flip
- [Compositing](docs/guides/features/compositing.md) — layer masks, transitions, fill operations
- [GPU Compute](docs/guides/features/gpu-compute.md) — wgpu shaders, pipeline caching, async readback
- [Pixel Buffers](docs/guides/features/pixel-buffers.md) — formats, views, pool, conversion

### Reference
- [API Reference](https://docs.rs/ranga) — rustdoc for all public items
- [Threat Model](docs/development/threat-model.md) — security trust boundaries
- Architecture Decisions: [001](docs/decisions/001-pure-rust-core.md), [002](docs/decisions/002-semver-versioning.md), [003](docs/decisions/003-packed-u32-gpu-shaders.md), [004](docs/decisions/004-feature-gated-optional-deps.md)

## Minimum Supported Rust Version

Rust **1.89** (edition 2024).

## License

[AGPL-3.0-only](LICENSE)
