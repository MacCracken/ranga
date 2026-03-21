# Compositing Guide

## Layer Compositing

The core operation for rasa and aethersafta — place a source layer onto a
destination canvas at a position with opacity:

```rust
use ranga::composite;

let mut canvas = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
composite::composite_at(&layer, &mut canvas, x, y, opacity)?;
```

Handles:
- Per-pixel alpha from the source
- Per-layer opacity (0.0–1.0)
- Automatic clipping to destination bounds
- Negative positions (partial visibility)
- Fast path for fully opaque and fully transparent pixels

## Alpha Operations

### Premultiplied Alpha
Some compositing pipelines (aethersafta) use premultiplied alpha internally:

```rust
composite::premultiply_alpha(&mut buf)?;   // straight → premultiplied
composite::unpremultiply_alpha(&mut buf)?; // premultiplied → straight
```

Roundtrip is within +/-1 due to integer division.

### Layer Masks
Apply a grayscale mask to control per-pixel transparency:

```rust
composite::apply_mask(&mut layer, &mask)?; // layer.alpha *= mask.red
```

The mask's red channel is used as the mask value (0 = transparent, 255 = opaque).

## Transitions

Replaces tazama's GPU transition shaders with CPU implementations:

### Dissolve (Cross-fade)
```rust
let mid = composite::dissolve(&clip_a, &clip_b, 0.5)?; // 50% mix
```

### Fade
```rust
composite::fade(&mut buf, 0.3)?; // 30% visible, RGB multiplied
```

### Wipe
```rust
let result = composite::wipe(&clip_a, &clip_b, 0.5)?; // left-to-right
```

## Fill Operations

### Solid Color
```rust
composite::fill_solid(&mut buf, [255, 0, 0, 255])?;
```

### Linear Gradient
```rust
composite::gradient_linear(&mut buf, [255, 0, 0, 255], [0, 0, 255, 255])?;
```
Runs left-to-right. Replaces rasa's `gradient_linear()`.

### Checkerboard
Standard transparency visualization pattern:
```rust
composite::fill_checkerboard(&mut buf, 16, [200, 200, 200, 255], [255, 255, 255, 255])?;
```
