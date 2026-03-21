# Performance Guide

## Feature Flags

Enable `simd` (default) for SIMD-accelerated blending. Enable `parallel` for
rayon-parallelized blur operations. Enable `gpu` for wgpu compute shaders on
large images.

## CPU vs GPU Crossover

GPU dispatch has fixed overhead (~5 ms for upload + pipeline + readback). Use
CPU for images below ~512x512 and GPU for 4K+. The `hwaccel` feature provides
`should_use_gpu(w, h)` to automate this decision.

| Image Size | Recommended |
|------------|-------------|
| < 256x256 | Always CPU |
| 256x256 – 1080p | CPU (SIMD blend, scalar filters) |
| 1080p – 4K | Profile both; GPU wins when pipeline is cached |
| 4K+ | GPU with pipeline caching |

## SIMD Blend Performance

`blend_row_normal` dispatches automatically:

| Architecture | Path | Pixels/iter | Speedup |
|---|---|---|---|
| x86_64 + AVX2 | AVX2 (runtime detected) | 4 | ~2x |
| x86_64 | SSE2 (baseline) | 2 | ~1.5x |
| aarch64 | NEON (baseline) | 8 | ~2x |
| Other | Scalar fallback | 1 | 1x |

Disable SIMD with `--no-default-features` for scalar-only builds.

## Filter Performance Tips

- **LUT-based filters** (levels, curves) are fastest — O(1) per pixel via lookup table
- **Gaussian blur** is O(n * radius) — use box blur for a fast approximation
- **Unsharp mask** calls Gaussian blur internally — cost is dominated by the blur
- **Hue shift** converts to HSL per pixel — expensive due to trig; batch if possible
- **3D LUT** with size 17 is the sweet spot (standard .cube size, good interpolation)

## Buffer Reuse

Use `BufferPool` in frame pipelines to avoid repeated allocation:

```rust
let mut pool = ranga::pixel::BufferPool::new(4);
for frame in frames {
    let buf = pool.acquire(w * h * 4);
    // ... process frame ...
    pool.release(buf);
}
```

## Zero-Copy Views

Use `PixelView` / `PixelViewMut` to pass existing buffers without cloning:

```rust
let view = ranga::pixel::PixelView::new(&existing_data, w, h, format)?;
```

## Benchmarking

Run all benchmarks:
```sh
cargo bench                        # default features (SIMD on)
cargo bench --no-default-features  # scalar only
cargo bench --all-features         # GPU + parallel + SIMD
cargo bench --bench filter         # single suite
```

Benchmark results are saved to `target/criterion/` with HTML reports.
