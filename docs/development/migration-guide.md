# Migration Guide: Adopting ranga in AGNOS Projects

## rasa (image editor)

### Color math (rasa-core/src/color.rs)
Replace inline implementations with ranga equivalents:

| rasa | ranga |
|------|-------|
| `srgb_to_linear()` / `linear_to_srgb()` | `ranga::color::srgb_to_linear()` / `linear_to_srgb()` |
| `Color::to_hsl()` / `Color::from_hsl()` | `ranga::color::Hsl` with `From<Srgba>` / `From<Hsl>` |
| `rgb_to_cmyk_naive()` / `cmyk_to_rgb_naive()` | `ranga::color::srgb_to_cmyk()` / `cmyk_to_srgb()` |
| `ColorSpace` enum | `ranga::color::ColorSpace` |
| `IccProfile` (lcms2-backed) | `ranga::icc::IccProfile` (pure Rust, matrix profiles) |

### Blend modes (rasa-core/src/blend.rs)
Replace `blend()` with `ranga::blend::blend_pixel()`. Same 12 modes, same Porter-Duff compositing.

### Filters (rasa-engine/src/filters.rs)
| rasa | ranga |
|------|-------|
| `apply_brightness_contrast()` | `ranga::filter::brightness()` + `ranga::filter::contrast()` |
| `apply_hue_saturation()` | `ranga::filter::hue_shift()` + `ranga::filter::saturation()` |
| `apply_curves()` | `ranga::filter::curves()` |
| `apply_levels()` | `ranga::filter::levels()` |
| `gaussian_blur()` | `ranga::filter::gaussian_blur()` |
| `sharpen()` | `ranga::filter::unsharp_mask()` |
| `invert()` / `grayscale()` | `ranga::filter::invert()` / `grayscale()` |

### GPU shaders (rasa-gpu/)
Replace wgpu pipeline setup and WGSL shaders with `ranga::gpu`:
- `GpuDevice` → `ranga::gpu::GpuContext`
- `dispatch_pixel_shader()` → `ranga::gpu::gpu_invert()`, `gpu_grayscale()`, etc.
- `dispatch_composite_shader()` → `ranga::gpu::gpu_blend()`

### Zero-copy integration
Use `PixelView` to pass rasa's existing buffers without copying:
```rust
let view = ranga::pixel::PixelView::new(&existing_bytes, w, h, PixelFormat::Rgba8)?;
```

## tazama (video editor)

### Color conversion (tazama/crates/media/src/convert.rs)
Replace `tarang::video::convert::yuv420p_to_rgb24()` chain with direct ranga calls:
- `ranga::convert::yuv420p_to_rgba()` (BT.601)
- `ranga::convert::yuv420p_to_rgba_bt709()` (BT.709 for HD)
- `ranga::convert::nv12_to_rgba()` (semi-planar)

### Color grading (tazama/crates/gpu/shaders/color_grade.comp)
Replace the Vulkan compute shader with ranga's wgpu pipeline:
- Brightness/contrast/saturation → `ranga::gpu::gpu_brightness_contrast()` + `gpu_saturation()`
- Or use CPU: `ranga::filter::brightness()` + `contrast()` + `saturation()`
- Color temperature → `ranga::color::color_temperature()` (apply multipliers)

### 3D LUT (tazama/crates/gpu/src/lut.rs)
Replace custom .cube parser with `ranga::filter::Lut3d::from_cube()` + `apply_lut3d()`.
Same trilinear interpolation algorithm.

### Buffer management
Use `BufferPool` for frame processing:
```rust
let mut pool = ranga::pixel::BufferPool::new(8);
// Per frame:
let buf = pool.acquire(w * h * 4);
// ... process ...
pool.release(buf);
```

## aethersafta (compositor)

### Blend/composite
Replace custom blend code with `ranga::blend::blend_pixel()` or `blend_row_normal()`.
SIMD-accelerated on x86_64 (SSE2/AVX2) and aarch64 (NEON).

### ARGB handling
aethersafta uses ARGB8 internally. Convert at boundaries:
- `ranga::convert::argb8_to_rgba8()` → process with ranga → `rgba8_to_argb8()`
- Or use `PixelView` for zero-copy reads

### GPU compositing
Replace custom GPU blend with `ranga::gpu::gpu_blend()` — supports all 12 modes
in a single parameterized shader.

## Feature flags

| Feature | What it enables |
|---------|----------------|
| `simd` (default) | SSE2/AVX2/NEON blend acceleration |
| `gpu` | wgpu compute shaders |
| `hwaccel` | GPU detection via ai-hwaccel |
| `parallel` | rayon row-parallel blur/filters |
