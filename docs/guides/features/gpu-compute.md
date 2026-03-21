# GPU Compute Guide

## Setup

Enable the `gpu` feature:
```toml
ranga = { version = "0.20", features = ["gpu"] }
```

Create a context (one-time, reuse for all operations):
```rust,no_run
use ranga::gpu::GpuContext;

let ctx = GpuContext::new()?;
println!("GPU: {} ({})", ctx.adapter_name(), ctx.backend_name());
```

## Available GPU Operations

| Function | CPU equivalent | Workgroup |
|----------|---------------|-----------|
| `gpu_blend` | `blend_row_normal` | 256 |
| `gpu_invert` | `filter::invert` | 256 |
| `gpu_grayscale` | `filter::grayscale` | 256 |
| `gpu_brightness_contrast` | `filter::brightness` + `contrast` | 256 |
| `gpu_saturation` | `filter::saturation` | 256 |
| `gpu_gaussian_blur` | `filter::gaussian_blur` | 16x16 |

All GPU functions accept `PixelBuffer` (RGBA8) and handle upload/download
internally.

## Pipeline Caching

`GpuContext` caches compiled compute pipelines. The first call to each shader
incurs compilation overhead (~5ms). Subsequent calls reuse the cached pipeline.

```rust,no_run
// First call: compiles shader
gpu_invert(&ctx, &mut buf1)?;  // ~5ms
// Second call: reuses pipeline
gpu_invert(&ctx, &mut buf2)?;  // ~1ms (upload + dispatch + download)
```

## Async Readback

For non-blocking GPU downloads:

```rust,no_run
let gpu_buf = GpuBuffer::upload(&ctx, &buf);
// ... dispatch shaders ...
let receiver = gpu_buf.download_async(&ctx);
// Do other CPU work here
let result = receiver.recv()??; // block when ready
```

## When to Use GPU

| Image Size | Recommendation |
|------------|---------------|
| < 512x512 | CPU always faster |
| 512x512 – 1080p | CPU usually faster (upload overhead) |
| 1080p – 4K | Profile; GPU wins with cached pipelines |
| 4K+ | GPU preferred |

Use `hwaccel::should_use_gpu(w, h)` for automatic recommendation.

## Hardware Detection

With the `hwaccel` feature:

```rust,no_run
let report = ranga::hwaccel::probe();
if report.has_gpu {
    println!("{} with {} MB", report.gpu_name, report.gpu_memory_mb);
}
```

## Shader Architecture

All WGSL shaders operate on packed u32 RGBA (not f32), avoiding the CPU-side
format conversion that rasa's GPU pipeline requires. Each shader includes
pack/unpack helpers:

```wgsl
fn unpack_rgba(packed: u32) -> vec4<f32> { ... }
fn pack_rgba(c: vec4<f32>) -> u32 { ... }
```

Blend shader supports all 12 modes via a `mode` parameter in the uniform buffer.
