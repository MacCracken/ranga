# Filters Guide

## Filter Inventory (23 total)

### Adjustment Filters (in-place)
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `brightness` | offset: -1.0..1.0 | Additive brightness shift |
| `contrast` | factor: 0.0+ | Scale around midpoint (1.0 = identity) |
| `saturation` | factor: 0.0+ | BT.601 luminance-based (0.0 = gray) |
| `vibrance` | amount: -1.0..1.0+ | Smart saturation (protects saturated colors) |
| `levels` | black, white, gamma | Input range remap with gamma curve |
| `curves` | LUT: [u8; 256] | Arbitrary per-channel lookup table |
| `hue_shift` | degrees | Rotate hue in HSL space |
| `color_balance` | shadows/mid/highlights | Per-range RGB offset |
| `channel_mixer` | 3x3 matrix | Arbitrary RGB channel weighting |
| `grayscale` | — | BT.601 luminance conversion |
| `invert` | — | 255 - value per channel |
| `threshold` | level: u8 | Binary black/white based on luminance |

### Spatial Filters (return new buffer)
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `gaussian_blur` | radius | Separable 2-pass, sigma = radius/3 |
| `box_blur` | radius | Uniform kernel (faster than Gaussian) |
| `unsharp_mask` | radius, amount | Sharpen = orig + amount × (orig - blur) |
| `median` | radius | Non-linear noise reduction |
| `bilateral` | radius, sigma_s, sigma_c | Edge-preserving smoothing |
| `vignette` | strength | Radial darkening from center |

### Noise (in-place, deterministic)
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `noise_gaussian` | amount, seed | Additive Gaussian noise |
| `noise_salt_pepper` | density, seed | Random black/white pixels |

### Color Grading
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `apply_lut3d` | Lut3d | .cube format 3D LUT with trilinear interpolation |

### Painting
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `flood_fill` | x, y, color, tolerance | 4-connected fill from seed point |

## Usage Patterns

### Basic adjustment pipeline
```rust
filter::brightness(&mut buf, 0.1)?;
filter::contrast(&mut buf, 1.2)?;
filter::saturation(&mut buf, 1.1)?;
```

### Noise reduction pipeline
```rust
// Remove salt-and-pepper noise with median, then smooth with bilateral
let denoised = filter::median(&noisy_buf, 1)?;
let smooth = filter::bilateral(&denoised, 2, 10.0, 30.0)?;
```

### Color grading with 3D LUT
```rust
let lut = filter::Lut3d::from_cube(include_str!("film_look.cube"))?;
filter::apply_lut3d(&mut buf, &lut)?;
```

### Sharpening after resize
```rust
let resized = transform::resize(&buf, 1920, 1080, ScaleFilter::Bilinear)?;
let sharp = filter::unsharp_mask(&resized, 1, 0.5)?;
```

## Performance Notes

- LUT-based filters (levels, curves) are fastest: O(1) per pixel via lookup
- Median and bilateral are O(n × radius²) — use smaller images or lower radius
- Gaussian blur is O(n × radius) per pass (2 passes for separable)
- With `parallel` feature, blur operations use rayon for row-level parallelism
- GPU variants available for blur, grayscale, invert, brightness/contrast, saturation
