//! Compositing operations — layer masks, transitions, gradients, alpha handling.
//!
//! Replaces inline compositing code in rasa (layer masks, gradient fill) and
//! tazama (dissolve/fade/wipe transitions).

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};

fn validate_rgba8(buf: &PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    Ok(())
}

fn validate_same_size(a: &PixelBuffer, b: &PixelBuffer) -> Result<(), RangaError> {
    if a.width != b.width || a.height != b.height {
        return Err(RangaError::DimensionMismatch {
            expected: a.pixel_count(),
            actual: b.pixel_count(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Alpha operations
// ---------------------------------------------------------------------------

/// Convert straight alpha to premultiplied alpha in-place.
///
/// `R = R * A / 255`, etc. Required for correct Porter-Duff compositing
/// in some pipelines (aethersafta uses premultiplied internally).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::new(vec![255, 128, 64, 128], 1, 1, PixelFormat::Rgba8).unwrap();
/// composite::premultiply_alpha(&mut buf).unwrap();
/// assert_eq!(buf.data[0], 128); // 255 * 128 / 255 ≈ 128
/// ```
pub fn premultiply_alpha(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    for pixel in buf.data.chunks_exact_mut(4) {
        let a = pixel[3] as u16;
        pixel[0] = ((pixel[0] as u16 * a) / 255) as u8;
        pixel[1] = ((pixel[1] as u16 * a) / 255) as u8;
        pixel[2] = ((pixel[2] as u16 * a) / 255) as u8;
    }
    Ok(())
}

/// Convert premultiplied alpha back to straight alpha in-place.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::new(vec![128, 64, 32, 128], 1, 1, PixelFormat::Rgba8).unwrap();
/// composite::unpremultiply_alpha(&mut buf).unwrap();
/// // 128 / (128/255) ≈ 255
/// assert!(buf.data[0] > 250);
/// ```
pub fn unpremultiply_alpha(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    for pixel in buf.data.chunks_exact_mut(4) {
        let a = pixel[3] as u16;
        if a == 0 {
            continue;
        }
        pixel[0] = ((pixel[0] as u16 * 255) / a).min(255) as u8;
        pixel[1] = ((pixel[1] as u16 * 255) / a).min(255) as u8;
        pixel[2] = ((pixel[2] as u16 * 255) / a).min(255) as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer mask
// ---------------------------------------------------------------------------

/// Apply an alpha mask to a buffer in-place.
///
/// The mask is a single-channel grayscale value per pixel. Each pixel's alpha
/// is multiplied by the corresponding mask value (0 = transparent, 255 = opaque).
/// The mask buffer must be RGBA8 with the same dimensions; the red channel is
/// used as the mask value.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let mask = PixelBuffer::new(vec![128, 128, 128, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// composite::apply_mask(&mut buf, &mask).unwrap();
/// assert_eq!(buf.data[3], 128); // alpha halved by mask
/// ```
pub fn apply_mask(buf: &mut PixelBuffer, mask: &PixelBuffer) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    validate_rgba8(mask)?;
    validate_same_size(buf, mask)?;
    for (pixel, mask_pixel) in buf.data.chunks_exact_mut(4).zip(mask.data.chunks_exact(4)) {
        let mask_val = mask_pixel[0] as u16; // use red channel as mask
        pixel[3] = ((pixel[3] as u16 * mask_val) / 255) as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Transitions (replaces tazama dissolve.comp, fade.comp, wipe.comp)
// ---------------------------------------------------------------------------

/// Dissolve (cross-fade) between two buffers.
///
/// `progress` is 0.0 (fully `a`) to 1.0 (fully `b`). Linear interpolation
/// per pixel. Replaces tazama's `dissolve.comp` shader.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let a = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let b = PixelBuffer::new(vec![0, 0, 255, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let mid = composite::dissolve(&a, &b, 0.5).unwrap();
/// assert!(mid.data[0] > 100 && mid.data[0] < 155); // ~128
/// assert!(mid.data[2] > 100 && mid.data[2] < 155);
/// ```
pub fn dissolve(
    a: &PixelBuffer,
    b: &PixelBuffer,
    progress: f32,
) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(a)?;
    validate_rgba8(b)?;
    validate_same_size(a, b)?;
    let t = progress.clamp(0.0, 1.0);
    let inv_t = 1.0 - t;
    let mut out = vec![0u8; a.data.len()];
    for (i, (pa, pb)) in a.data.iter().zip(b.data.iter()).enumerate() {
        out[i] = (*pa as f32 * inv_t + *pb as f32 * t + 0.5).clamp(0.0, 255.0) as u8;
    }
    PixelBuffer::new(out, a.width, a.height, PixelFormat::Rgba8)
}

/// Fade a buffer by multiplying RGB by `progress` (0.0 = black, 1.0 = full).
///
/// Replaces tazama's `fade.comp` shader. Alpha is preserved.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::new(vec![200, 200, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// composite::fade(&mut buf, 0.5).unwrap();
/// assert_eq!(buf.data[0], 100);
/// ```
pub fn fade(buf: &mut PixelBuffer, progress: f32) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    let p = progress.clamp(0.0, 1.0);
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = (pixel[0] as f32 * p + 0.5).clamp(0.0, 255.0) as u8;
        pixel[1] = (pixel[1] as f32 * p + 0.5).clamp(0.0, 255.0) as u8;
        pixel[2] = (pixel[2] as f32 * p + 0.5).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Hard wipe transition (left-to-right) between two buffers.
///
/// `progress` is 0.0 (fully `a`) to 1.0 (fully `b`). Replaces tazama's
/// `wipe.comp` shader.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let a = PixelBuffer::new(vec![255; 8 * 1 * 4], 8, 1, PixelFormat::Rgba8).unwrap();
/// let b = PixelBuffer::new(vec![0; 8 * 1 * 4], 8, 1, PixelFormat::Rgba8).unwrap();
/// let result = composite::wipe(&a, &b, 0.5).unwrap();
/// assert_eq!(result.data[0], 0);    // left half is b (black)
/// assert_eq!(result.data[7 * 4], 255); // right half is a (white)
/// ```
pub fn wipe(a: &PixelBuffer, b: &PixelBuffer, progress: f32) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(a)?;
    validate_rgba8(b)?;
    validate_same_size(a, b)?;
    let threshold = (progress.clamp(0.0, 1.0) * a.width as f32) as u32;
    let w = a.width as usize;
    let mut out = vec![0u8; a.data.len()];
    for y in 0..a.height as usize {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let src = if (x as u32) < threshold {
                &b.data
            } else {
                &a.data
            };
            out[i..i + 4].copy_from_slice(&src[i..i + 4]);
        }
    }
    PixelBuffer::new(out, a.width, a.height, PixelFormat::Rgba8)
}

// ---------------------------------------------------------------------------
// Fill operations
// ---------------------------------------------------------------------------

/// Fill a buffer with a solid RGBA color.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
/// composite::fill_solid(&mut buf, [255, 0, 0, 255]).unwrap();
/// assert_eq!(buf.data[0], 255);
/// ```
pub fn fill_solid(buf: &mut PixelBuffer, color: [u8; 4]) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel.copy_from_slice(&color);
    }
    Ok(())
}

/// Fill a buffer with a linear gradient from `start` to `end` color.
///
/// Gradient runs left-to-right. Replaces rasa's `gradient_linear`.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::zeroed(100, 1, PixelFormat::Rgba8);
/// composite::gradient_linear(&mut buf, [255, 0, 0, 255], [0, 0, 255, 255]).unwrap();
/// assert!(buf.data[0] > 200); // left = red
/// assert!(buf.data[99 * 4 + 2] > 200); // right = blue
/// ```
pub fn gradient_linear(
    buf: &mut PixelBuffer,
    start: [u8; 4],
    end: [u8; 4],
) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    let w = buf.width as f32;
    if w < 1.0 {
        return Ok(());
    }
    for y in 0..buf.height as usize {
        for x in 0..buf.width as usize {
            let t = x as f32 / (w - 1.0).max(1.0);
            let i = (y * buf.width as usize + x) * 4;
            for c in 0..4 {
                buf.data[i + c] =
                    (start[c] as f32 + t * (end[c] as f32 - start[c] as f32) + 0.5) as u8;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Positioned composite (core aethersafta operation)
// ---------------------------------------------------------------------------

/// Composite `src` onto `dst` at position `(x, y)` with the given opacity.
///
/// Performs Porter-Duff "source over" with per-pixel alpha and per-layer
/// opacity. Pixels outside `dst` bounds are clipped. Both buffers must be
/// RGBA8. This is the core operation that replaces aethersafta's
/// `compositor::blend_frame()`.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut dst = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
/// let src = PixelBuffer::new(vec![255, 0, 0, 200].repeat(16), 4, 4, PixelFormat::Rgba8).unwrap();
/// composite::composite_at(&src, &mut dst, 10, 10, 1.0).unwrap();
/// // Pixel at (10,10) should now have red content
/// assert!(dst.data[(10 * 100 + 10) * 4] > 0);
/// ```
pub fn composite_at(
    src: &PixelBuffer,
    dst: &mut PixelBuffer,
    x: i32,
    y: i32,
    opacity: f32,
) -> Result<(), RangaError> {
    validate_rgba8(src)?;
    validate_rgba8(dst)?;

    let dw = dst.width as i32;
    let dh = dst.height as i32;
    let sw = src.width as i32;
    let sh = src.height as i32;

    // Clip to destination bounds
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + sw).min(dw);
    let y1 = (y + sh).min(dh);
    if x0 >= x1 || y0 >= y1 {
        return Ok(()); // fully clipped
    }

    let op = (opacity.clamp(0.0, 1.0) * 255.0) as u16;
    if op == 0 {
        return Ok(());
    }

    let dst_stride = dst.width as usize;
    let src_stride = src.width as usize;

    for dy in y0..y1 {
        let sy = (dy - y) as usize;
        for dx in x0..x1 {
            let sx = (dx - x) as usize;
            let si = (sy * src_stride + sx) * 4;
            let di = (dy as usize * dst_stride + dx as usize) * 4;

            let sa = (src.data[si + 3] as u16 * op) >> 8;
            if sa == 0 {
                continue;
            }
            if sa >= 255 {
                dst.data[di..di + 4].copy_from_slice(&src.data[si..si + 4]);
                continue;
            }
            let inv_sa = 255u16 - sa;
            dst.data[di] = ((src.data[si] as u16 * sa + dst.data[di] as u16 * inv_sa) >> 8) as u8;
            dst.data[di + 1] =
                ((src.data[si + 1] as u16 * sa + dst.data[di + 1] as u16 * inv_sa) >> 8) as u8;
            dst.data[di + 2] =
                ((src.data[si + 2] as u16 * sa + dst.data[di + 2] as u16 * inv_sa) >> 8) as u8;
            dst.data[di + 3] = ((sa + dst.data[di + 3] as u16 * inv_sa / 255).min(255)) as u8;
        }
    }
    Ok(())
}

/// Composite `src` onto `dst` at position `(x, y)` — ARGB8 variant.
///
/// Same as [`composite_at`] but for ARGB8 pixel layout (`[A, R, G, B]`)
/// used by aethersafta. Both buffers must be Argb8 format.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Argb8);
/// let src = PixelBuffer::new(vec![200, 255, 0, 0].repeat(4), 2, 2, PixelFormat::Argb8).unwrap();
/// composite::composite_at_argb(&src, &mut dst, 1, 1, 1.0).unwrap();
/// ```
pub fn composite_at_argb(
    src: &PixelBuffer,
    dst: &mut PixelBuffer,
    x: i32,
    y: i32,
    opacity: f32,
) -> Result<(), RangaError> {
    if src.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", src.format)));
    }
    if dst.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", dst.format)));
    }

    let dw = dst.width as i32;
    let dh = dst.height as i32;
    let sw = src.width as i32;
    let sh = src.height as i32;

    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + sw).min(dw);
    let y1 = (y + sh).min(dh);
    if x0 >= x1 || y0 >= y1 {
        return Ok(());
    }

    let op = (opacity.clamp(0.0, 1.0) * 255.0) as u16;
    if op == 0 {
        return Ok(());
    }

    let dst_stride = dst.width as usize;
    let src_stride = src.width as usize;

    for dy in y0..y1 {
        let sy = (dy - y) as usize;
        for dx in x0..x1 {
            let sx = (dx - x) as usize;
            let si = (sy * src_stride + sx) * 4;
            let di = (dy as usize * dst_stride + dx as usize) * 4;

            // ARGB layout: [A, R, G, B]
            let sa = (src.data[si] as u16 * op) >> 8;
            if sa == 0 {
                continue;
            }
            if sa >= 255 {
                dst.data[di..di + 4].copy_from_slice(&src.data[si..si + 4]);
                continue;
            }
            let inv_sa = 255u16 - sa;
            dst.data[di] = ((sa + dst.data[di] as u16 * inv_sa / 255).min(255)) as u8;
            dst.data[di + 1] =
                ((src.data[si + 1] as u16 * sa + dst.data[di + 1] as u16 * inv_sa) >> 8) as u8;
            dst.data[di + 2] =
                ((src.data[si + 2] as u16 * sa + dst.data[di + 2] as u16 * inv_sa) >> 8) as u8;
            dst.data[di + 3] =
                ((src.data[si + 3] as u16 * sa + dst.data[di + 3] as u16 * inv_sa) >> 8) as u8;
        }
    }
    Ok(())
}

/// Fill a checkerboard pattern (transparency visualization).
///
/// Alternating light/dark squares of `block_size` pixels. Standard pattern
/// used by image editors to indicate transparency.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::composite;
///
/// let mut buf = PixelBuffer::zeroed(16, 16, PixelFormat::Rgba8);
/// composite::fill_checkerboard(&mut buf, 8, [200, 200, 200, 255], [255, 255, 255, 255]).unwrap();
/// assert_ne!(buf.data[0..4], buf.data[8 * 4..8 * 4 + 4]); // alternating
/// ```
pub fn fill_checkerboard(
    buf: &mut PixelBuffer,
    block_size: u32,
    color_a: [u8; 4],
    color_b: [u8; 4],
) -> Result<(), RangaError> {
    validate_rgba8(buf)?;
    let bs = block_size.max(1) as usize;
    for y in 0..buf.height as usize {
        for x in 0..buf.width as usize {
            let i = (y * buf.width as usize + x) * 4;
            let color = if ((x / bs) + (y / bs)).is_multiple_of(2) {
                &color_a
            } else {
                &color_b
            };
            buf.data[i..i + 4].copy_from_slice(color);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_unpremultiply_roundtrip() {
        let mut buf = PixelBuffer::new(vec![200, 100, 50, 128], 1, 1, PixelFormat::Rgba8).unwrap();
        let original = buf.data.clone();
        premultiply_alpha(&mut buf).unwrap();
        assert_ne!(buf.data[0..3], original[0..3]);
        unpremultiply_alpha(&mut buf).unwrap();
        for (i, (&got, &exp)) in buf.data.iter().zip(original.iter()).enumerate().take(3) {
            assert!((got as i16 - exp as i16).unsigned_abs() <= 1, "channel {i}");
        }
    }

    #[test]
    fn premultiply_fully_opaque_unchanged() {
        let mut buf = PixelBuffer::new(vec![200, 100, 50, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let original = buf.data.clone();
        premultiply_alpha(&mut buf).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn premultiply_fully_transparent_zeroes() {
        let mut buf = PixelBuffer::new(vec![200, 100, 50, 0], 1, 1, PixelFormat::Rgba8).unwrap();
        premultiply_alpha(&mut buf).unwrap();
        assert_eq!(buf.data[0], 0);
        assert_eq!(buf.data[1], 0);
        assert_eq!(buf.data[2], 0);
    }

    #[test]
    fn apply_mask_half_alpha() {
        let mut buf = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let mask = PixelBuffer::new(vec![128, 128, 128, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        apply_mask(&mut buf, &mask).unwrap();
        assert_eq!(buf.data[3], 128);
    }

    #[test]
    fn dissolve_endpoints() {
        let a = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let b = PixelBuffer::new(vec![0, 255, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let at_0 = dissolve(&a, &b, 0.0).unwrap();
        let at_1 = dissolve(&a, &b, 1.0).unwrap();
        assert_eq!(at_0.data, a.data);
        assert_eq!(at_1.data, b.data);
    }

    #[test]
    fn dissolve_midpoint() {
        let a = PixelBuffer::new(vec![200, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let b = PixelBuffer::new(vec![0, 200, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let mid = dissolve(&a, &b, 0.5).unwrap();
        assert!((mid.data[0] as i16 - 100).abs() <= 1);
        assert!((mid.data[1] as i16 - 100).abs() <= 1);
    }

    #[test]
    fn fade_zero_is_black() {
        let mut buf = PixelBuffer::new(vec![200, 200, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        fade(&mut buf, 0.0).unwrap();
        assert_eq!(buf.data[0], 0);
        assert_eq!(buf.data[3], 255); // alpha preserved
    }

    #[test]
    fn wipe_half() {
        let a = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let b = PixelBuffer::new(vec![0; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let result = wipe(&a, &b, 0.5).unwrap();
        assert_eq!(result.data[0], 0); // left half = b
        assert_eq!(result.data[3 * 4], 255); // right half = a
    }

    #[test]
    fn fill_solid_works() {
        let mut buf = PixelBuffer::zeroed(2, 2, PixelFormat::Rgba8);
        fill_solid(&mut buf, [255, 128, 64, 255]).unwrap();
        assert_eq!(buf.data[0], 255);
        assert_eq!(buf.data[5], 128); // second pixel G
    }

    #[test]
    fn gradient_endpoints() {
        let mut buf = PixelBuffer::zeroed(10, 1, PixelFormat::Rgba8);
        gradient_linear(&mut buf, [255, 0, 0, 255], [0, 0, 255, 255]).unwrap();
        assert!(buf.data[0] > 200); // left = mostly red
        assert!(buf.data[9 * 4 + 2] > 200); // right = mostly blue
    }

    #[test]
    fn composite_at_basic() {
        let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
        let src = PixelBuffer::new([255, 0, 0, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        composite_at(&src, &mut dst, 3, 3, 1.0).unwrap();
        let i = (3 * 10 + 3) * 4;
        assert!(dst.data[i] > 200); // red pixel at (3,3)
    }

    #[test]
    fn composite_at_clipped() {
        let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
        let src = PixelBuffer::new([255, 0, 0, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        // Place at edge — partially clipped
        composite_at(&src, &mut dst, 9, 9, 1.0).unwrap();
        let i = (9 * 10 + 9) * 4;
        assert!(dst.data[i] > 200); // one pixel visible
    }

    #[test]
    fn composite_at_negative_pos() {
        let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
        let src = PixelBuffer::new([255, 0, 0, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        composite_at(&src, &mut dst, -1, -1, 1.0).unwrap();
        // Only pixel (0,0) of dst gets the bottom-right pixel of src
        assert!(dst.data[0] > 200);
    }

    #[test]
    fn composite_at_zero_opacity() {
        let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
        let src = PixelBuffer::new([255, 0, 0, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        composite_at(&src, &mut dst, 0, 0, 0.0).unwrap();
        assert_eq!(dst.data[0], 0); // unchanged
    }

    #[test]
    fn checkerboard_alternates() {
        let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        fill_checkerboard(&mut buf, 2, [100, 100, 100, 255], [200, 200, 200, 255]).unwrap();
        assert_eq!(buf.data[0], 100); // top-left block = color_a
        assert_eq!(buf.data[2 * 4], 200); // (2,0) = color_b
    }
}
