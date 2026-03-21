# Pixel Buffers Guide

## PixelBuffer

The core data type — a validated byte buffer with known format and dimensions:

```rust
use ranga::pixel::{PixelBuffer, PixelFormat};

let buf = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
let buf = PixelBuffer::new(data, 1920, 1080, PixelFormat::Rgba8)?;
```

`new()` validates that `data.len()` matches the expected size for the format.

## Supported Formats

| Format | Bytes/pixel | Description |
|--------|------------|-------------|
| `Rgba8` | 4 | Standard 8-bit RGBA (primary format for all operations) |
| `Argb8` | 4 | Alpha-first (used by aethersafta) |
| `Rgb8` | 3 | No alpha channel |
| `Yuv420p` | 1.5 | Planar YUV 4:2:0 (video) |
| `Nv12` | 1.5 | Semi-planar YUV 4:2:0 (video) |
| `RgbaF32` | 16 | 32-bit float per channel (HDR) |

## Pixel Access

```rust
// Typed accessors (RGBA8 only)
let pixel = buf.get_rgba(10, 20);        // Option<[u8; 4]>
buf.set_rgba(10, 20, [255, 0, 0, 255]);  // returns bool

// Row iteration
for row in buf.rows() {
    // row: &[u8], one row of pixels
}
for row in buf.rows_mut() {
    row[0] = 255; // modify in place
}

// Raw data access
let byte = buf.data[0];
```

## Zero-Copy Views

Borrow existing buffers from rasa/tazama/aethersafta without copying:

```rust
use ranga::pixel::PixelView;

// From raw bytes (e.g., from another library's buffer)
let view = PixelView::new(&existing_bytes, w, h, PixelFormat::Rgba8)?;

// From PixelBuffer
let view = buf.as_view();
let mut view = buf.as_view_mut();

// Convert back to owned
let owned = PixelBuffer::from_view(&view);
```

## Buffer Pool

Reuse allocations in frame-processing pipelines:

```rust
use ranga::pixel::BufferPool;

let mut pool = BufferPool::new(4); // keep up to 4 buffers

loop {
    let buf = pool.acquire(1920 * 1080 * 4); // zero-filled
    // ... process frame ...
    pool.release(buf); // return to pool
}
```

`acquire()` reuses a pooled buffer if one of sufficient capacity exists,
otherwise allocates new. Returned buffers are always zero-filled.

## Format Conversion

Convert between formats using the `convert` module:

```rust
use ranga::convert;

let rgba = convert::rgb8_to_rgba8(&rgb_buf)?;      // add alpha
let rgb = convert::rgba8_to_rgb8(&rgba_buf)?;       // strip alpha
let rgba = convert::argb8_to_rgba8(&argb_buf)?;     // reorder channels
let yuv = convert::rgba_to_yuv420p(&rgba_buf)?;     // BT.601
let yuv = convert::rgba_to_yuv420p_bt709(&rgba_buf)?; // BT.709
let nv12 = convert::argb_to_nv12(&argb_buf)?;       // semi-planar
```
