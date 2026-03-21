# Transforms Guide

## Operations

### Crop
Extract a rectangular region:
```rust
let cropped = transform::crop(&buf, 10, 20, 200, 150)?;
```
Coordinates are clamped to image bounds. Inverted rectangles produce empty output.

### Resize
Scale to new dimensions:
```rust
let small = transform::resize(&buf, 640, 480, ScaleFilter::Bilinear)?;
let thumb = transform::resize(&buf, 128, 128, ScaleFilter::Nearest)?;
```

| Filter | Quality | Speed | Use case |
|--------|---------|-------|----------|
| Nearest | Blocky | Fast | Thumbnails, pixel art |
| Bilinear | Smooth | Medium | General purpose |

### Flip
Mirror horizontally or vertically:
```rust
let mirrored = transform::flip_horizontal(&buf)?;
let flipped = transform::flip_vertical(&buf)?;
```
Both are involutions — applying twice returns the original.

### Affine Transform
Arbitrary 2D affine (scale + rotate + translate):
```rust
use ranga::transform::{Affine, ScaleFilter};

let t = Affine::rotate(0.1)
    .then(&Affine::scale(1.5, 1.5))
    .then(&Affine::translate(100.0, 50.0));

let result = transform::affine_transform(&buf, &t, 1920, 1080, ScaleFilter::Bilinear)?;
```

Uses inverse mapping — each output pixel samples the source at the inverse-transformed
coordinate. Pixels outside the source are transparent black.

## Affine Matrix

The `Affine` type is a 3x2 matrix supporting:
- `Affine::translate(tx, ty)` — shift
- `Affine::scale(sx, sy)` — scale (non-uniform)
- `Affine::rotate(radians)` — rotation around origin
- `a.then(&b)` — compose (apply `b` first, then `a`)
- `a.inverse()` — compute inverse (returns `None` if singular)
- `a.apply(x, y)` — transform a point
- `a.is_identity()` — check if no-op

Replaces `rasa-core/transform.rs` with the same API.
