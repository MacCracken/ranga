//! Geometry transforms — crop, resize, affine, flip.
//!
//! Provides spatial operations on [`PixelBuffer`]s. Replaces inline transform
//! code in rasa (`rasa-core/transform.rs`) and tazama (`crop.comp`, `transform.comp`).

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Affine transform matrix
// ---------------------------------------------------------------------------

/// 2D affine transform matrix (3x2).
///
/// ```text
/// | a  c  tx |
/// | b  d  ty |
/// | 0  0  1  |
/// ```
///
/// # Examples
///
/// ```
/// use ranga::transform::Affine;
///
/// let t = Affine::translate(10.0, 0.0).then(&Affine::scale(2.0, 2.0));
/// let (x, y) = t.apply(5.0, 5.0);
/// assert!((x - 20.0).abs() < 1e-9); // scale first: 5*2=10, then translate: 10+10=20
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Affine {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Translation.
    #[must_use]
    pub fn translate(tx: f64, ty: f64) -> Self {
        Self {
            tx,
            ty,
            ..Self::IDENTITY
        }
    }

    /// Uniform or non-uniform scale.
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            d: sy,
            ..Self::IDENTITY
        }
    }

    /// Rotation by angle in radians.
    #[must_use]
    pub fn rotate(angle_rad: f64) -> Self {
        let (sin, cos) = angle_rad.sin_cos();
        Self {
            a: cos,
            b: sin,
            c: -sin,
            d: cos,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Compose: `self * other` (apply `other` first, then `self`).
    #[must_use]
    pub fn then(&self, other: &Affine) -> Self {
        Self {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            tx: self.a * other.tx + self.c * other.ty + self.tx,
            ty: self.b * other.tx + self.d * other.ty + self.ty,
        }
    }

    /// Apply transform to a point.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )
    }

    /// Compute the inverse transform, if non-singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self {
            a: self.d * inv,
            b: -self.b * inv,
            c: -self.c * inv,
            d: self.a * inv,
            tx: (self.c * self.ty - self.d * self.tx) * inv,
            ty: (self.b * self.tx - self.a * self.ty) * inv,
        })
    }

    /// Check if this is the identity transform.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.a - 1.0).abs() < 1e-9
            && self.b.abs() < 1e-9
            && self.c.abs() < 1e-9
            && (self.d - 1.0).abs() < 1e-9
            && self.tx.abs() < 1e-9
            && self.ty.abs() < 1e-9
    }
}

impl Default for Affine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ---------------------------------------------------------------------------
// Resize filter
// ---------------------------------------------------------------------------

/// Interpolation filter for resize operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScaleFilter {
    /// Nearest-neighbor (fastest, blocky).
    Nearest,
    /// Bilinear interpolation (smooth, good default).
    Bilinear,
}

// ---------------------------------------------------------------------------
// Pixel buffer operations
// ---------------------------------------------------------------------------

fn validate_rgba8(buf: &PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    Ok(())
}

/// Crop a rectangular region from an RGBA8 buffer.
///
/// Returns a new buffer with dimensions `(right - left) x (bottom - top)`.
/// Coordinates are clamped to image bounds.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::transform;
///
/// let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
/// let cropped = transform::crop(&buf, 10, 20, 50, 60).unwrap();
/// assert_eq!(cropped.width, 40);
/// assert_eq!(cropped.height, 40);
/// ```
pub fn crop(
    buf: &PixelBuffer,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(buf)?;
    let l = left.min(buf.width) as usize;
    let t = top.min(buf.height) as usize;
    let r = right.min(buf.width).max(l as u32) as usize;
    let b = bottom.min(buf.height).max(t as u32) as usize;
    let out_w = r - l;
    let out_h = b - t;
    if out_w == 0 || out_h == 0 {
        return PixelBuffer::new(vec![], 0, 0, PixelFormat::Rgba8)
            .or_else(|_| Ok(PixelBuffer::zeroed(0, 0, PixelFormat::Rgba8)));
    }
    let src_stride = buf.width as usize * 4;
    let mut out = vec![0u8; out_w * out_h * 4];
    for y in 0..out_h {
        let src_off = (t + y) * src_stride + l * 4;
        let dst_off = y * out_w * 4;
        out[dst_off..dst_off + out_w * 4].copy_from_slice(&buf.data[src_off..src_off + out_w * 4]);
    }
    PixelBuffer::new(out, out_w as u32, out_h as u32, PixelFormat::Rgba8)
}

/// Resize an RGBA8 buffer to new dimensions.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::transform::{self, ScaleFilter};
///
/// let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
/// let small = transform::resize(&buf, 50, 50, ScaleFilter::Bilinear).unwrap();
/// assert_eq!(small.width, 50);
/// ```
pub fn resize(
    buf: &PixelBuffer,
    new_w: u32,
    new_h: u32,
    filter: ScaleFilter,
) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(buf)?;
    if new_w == 0 || new_h == 0 {
        return Ok(PixelBuffer::zeroed(0, 0, PixelFormat::Rgba8));
    }
    let sw = buf.width as usize;
    let sh = buf.height as usize;
    let dw = new_w as usize;
    let dh = new_h as usize;
    let mut out = vec![0u8; dw * dh * 4];

    match filter {
        ScaleFilter::Nearest => {
            for dy in 0..dh {
                let sy = (dy * sh / dh).min(sh - 1);
                for dx in 0..dw {
                    let sx = (dx * sw / dw).min(sw - 1);
                    let si = (sy * sw + sx) * 4;
                    let di = (dy * dw + dx) * 4;
                    out[di..di + 4].copy_from_slice(&buf.data[si..si + 4]);
                }
            }
        }
        ScaleFilter::Bilinear => {
            let x_ratio = if dw > 1 {
                (sw as f64 - 1.0) / (dw as f64 - 1.0)
            } else {
                0.0
            };
            let y_ratio = if dh > 1 {
                (sh as f64 - 1.0) / (dh as f64 - 1.0)
            } else {
                0.0
            };
            for dy in 0..dh {
                let sy = dy as f64 * y_ratio;
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(sh - 1);
                let fy = sy - y0 as f64;
                for dx in 0..dw {
                    let sx_f = dx as f64 * x_ratio;
                    let x0 = sx_f.floor() as usize;
                    let x1 = (x0 + 1).min(sw - 1);
                    let fx = sx_f - x0 as f64;
                    let di = (dy * dw + dx) * 4;
                    for c in 0..4 {
                        let c00 = buf.data[(y0 * sw + x0) * 4 + c] as f64;
                        let c10 = buf.data[(y0 * sw + x1) * 4 + c] as f64;
                        let c01 = buf.data[(y1 * sw + x0) * 4 + c] as f64;
                        let c11 = buf.data[(y1 * sw + x1) * 4 + c] as f64;
                        let top = c00 + fx * (c10 - c00);
                        let bot = c01 + fx * (c11 - c01);
                        let val = top + fy * (bot - top);
                        out[di + c] = val.clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    PixelBuffer::new(out, new_w, new_h, PixelFormat::Rgba8)
}

/// Flip an RGBA8 buffer horizontally (mirror left↔right).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::transform;
///
/// let buf = PixelBuffer::new(vec![255, 0, 0, 255, 0, 255, 0, 255], 2, 1, PixelFormat::Rgba8).unwrap();
/// let flipped = transform::flip_horizontal(&buf).unwrap();
/// assert_eq!(flipped.data[0], 0); // green pixel now first
/// ```
pub fn flip_horizontal(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(buf)?;
    let w = buf.width as usize;
    let h = buf.height as usize;
    let mut out = vec![0u8; buf.data.len()];
    for y in 0..h {
        for x in 0..w {
            let si = (y * w + x) * 4;
            let di = (y * w + (w - 1 - x)) * 4;
            out[di..di + 4].copy_from_slice(&buf.data[si..si + 4]);
        }
    }
    PixelBuffer::new(out, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Flip an RGBA8 buffer vertically (mirror top↔bottom).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::transform;
///
/// let mut data = vec![0u8; 2 * 2 * 4];
/// data[0] = 255; // top-left red
/// let buf = PixelBuffer::new(data, 2, 2, PixelFormat::Rgba8).unwrap();
/// let flipped = transform::flip_vertical(&buf).unwrap();
/// assert_eq!(flipped.data[2 * 4], 255); // now bottom-left
/// ```
pub fn flip_vertical(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(buf)?;
    let w = buf.width as usize;
    let h = buf.height as usize;
    let stride = w * 4;
    let mut out = vec![0u8; buf.data.len()];
    for y in 0..h {
        let src_off = y * stride;
        let dst_off = (h - 1 - y) * stride;
        out[dst_off..dst_off + stride].copy_from_slice(&buf.data[src_off..src_off + stride]);
    }
    PixelBuffer::new(out, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Apply an affine transform to an RGBA8 buffer with the given interpolation.
///
/// The output has dimensions `out_w x out_h`. Pixels outside the source are
/// transparent black. Uses inverse mapping (sample source at transformed coords).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::transform::{self, Affine, ScaleFilter};
///
/// let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
/// let rotated = transform::affine_transform(&buf, &Affine::rotate(0.5), 100, 100, ScaleFilter::Bilinear).unwrap();
/// assert_eq!(rotated.width, 100);
/// ```
pub fn affine_transform(
    buf: &PixelBuffer,
    transform: &Affine,
    out_w: u32,
    out_h: u32,
    filter: ScaleFilter,
) -> Result<PixelBuffer, RangaError> {
    validate_rgba8(buf)?;
    let inv = transform
        .inverse()
        .ok_or_else(|| RangaError::Other("singular affine transform cannot be inverted".into()))?;
    let sw = buf.width as usize;
    let sh = buf.height as usize;
    let dw = out_w as usize;
    let dh = out_h as usize;
    let mut out = vec![0u8; dw * dh * 4];

    for dy in 0..dh {
        for dx in 0..dw {
            let (sx, sy) = inv.apply(dx as f64 + 0.5, dy as f64 + 0.5);
            let di = (dy * dw + dx) * 4;

            if sx < 0.0 || sy < 0.0 || sx >= sw as f64 || sy >= sh as f64 {
                continue; // transparent black
            }

            match filter {
                ScaleFilter::Nearest => {
                    let ix = sx as usize;
                    let iy = sy as usize;
                    let si = (iy * sw + ix) * 4;
                    out[di..di + 4].copy_from_slice(&buf.data[si..si + 4]);
                }
                ScaleFilter::Bilinear => {
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(sw - 1);
                    let y1 = (y0 + 1).min(sh - 1);
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;
                    for c in 0..4 {
                        let c00 = buf.data[(y0 * sw + x0) * 4 + c] as f64;
                        let c10 = buf.data[(y0 * sw + x1) * 4 + c] as f64;
                        let c01 = buf.data[(y1 * sw + x0) * 4 + c] as f64;
                        let c11 = buf.data[(y1 * sw + x1) * 4 + c] as f64;
                        let top = c00 + fx * (c10 - c00);
                        let bot = c01 + fx * (c11 - c01);
                        out[di + c] = (top + fy * (bot - top)).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    PixelBuffer::new(out, out_w, out_h, PixelFormat::Rgba8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn affine_identity() {
        assert!(Affine::IDENTITY.is_identity());
    }

    #[test]
    fn affine_translate() {
        let (x, y) = Affine::translate(10.0, 20.0).apply(5.0, 5.0);
        assert!((x - 15.0).abs() < 1e-9);
        assert!((y - 25.0).abs() < 1e-9);
    }

    #[test]
    fn affine_scale() {
        let (x, y) = Affine::scale(2.0, 3.0).apply(5.0, 5.0);
        assert!((x - 10.0).abs() < 1e-9);
        assert!((y - 15.0).abs() < 1e-9);
    }

    #[test]
    fn affine_rotate_90() {
        let (x, y) = Affine::rotate(PI / 2.0).apply(1.0, 0.0);
        assert!(x.abs() < 1e-9);
        assert!((y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn affine_inverse_roundtrip() {
        let t = Affine::translate(10.0, 20.0)
            .then(&Affine::scale(2.0, 3.0))
            .then(&Affine::rotate(0.5));
        let inv = t.inverse().unwrap();
        let (x, y) = t.apply(7.0, 13.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 7.0).abs() < 1e-9);
        assert!((by - 13.0).abs() < 1e-9);
    }

    #[test]
    fn affine_singular_no_inverse() {
        assert!(Affine::scale(0.0, 1.0).inverse().is_none());
    }

    #[test]
    fn crop_basic() {
        let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
        let c = crop(&buf, 10, 20, 50, 60).unwrap();
        assert_eq!(c.width, 40);
        assert_eq!(c.height, 40);
    }

    #[test]
    fn crop_clamped() {
        let buf = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
        let c = crop(&buf, 0, 0, 999, 999).unwrap();
        assert_eq!(c.width, 10);
        assert_eq!(c.height, 10);
    }

    #[test]
    fn resize_nearest() {
        let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
        let r = resize(&buf, 50, 50, ScaleFilter::Nearest).unwrap();
        assert_eq!(r.width, 50);
    }

    #[test]
    fn resize_bilinear() {
        let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
        let r = resize(&buf, 200, 200, ScaleFilter::Bilinear).unwrap();
        assert_eq!(r.width, 200);
    }

    #[test]
    fn flip_h_roundtrip() {
        let buf = PixelBuffer::new(
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        let f1 = flip_horizontal(&buf).unwrap();
        let f2 = flip_horizontal(&f1).unwrap();
        assert_eq!(f2.data, buf.data);
    }

    #[test]
    fn flip_v_roundtrip() {
        let data: Vec<u8> = (0..16).collect();
        let buf = PixelBuffer::new(data.clone(), 2, 2, PixelFormat::Rgba8).unwrap();
        let f1 = flip_vertical(&buf).unwrap();
        let f2 = flip_vertical(&f1).unwrap();
        assert_eq!(f2.data, data);
    }
}
