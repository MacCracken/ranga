//! Pixel format conversion — RGB↔YUV, ARGB↔NV12, format interop.
//!
//! Uses fixed-point BT.601 and BT.709 coefficients for integer-only fast paths.
//! Replaces inline conversion code from rasa, tazama, and tarang with a single
//! shared implementation.

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};

// =========================================================================
// Y-plane computation helper
// =========================================================================

/// Compute BT.601 luminance for a row of RGBA8 pixels: Y = (77*R + 150*G + 29*B) >> 8
fn compute_y_row(rgba: &[u8], y_out: &mut [u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        compute_y_row_simd_sse2(rgba, y_out, 77, 150, 29);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline for aarch64.
        unsafe { compute_y_row_simd_neon(rgba, y_out, 77, 150, 29) };
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    compute_y_row_scalar(rgba, y_out, 77, 150, 29);
}

/// Compute BT.709 luminance for a row: Y = (54*R + 183*G + 18*B) >> 8
fn compute_y_row_bt709(rgba: &[u8], y_out: &mut [u8]) {
    // BT.709: 0.2126*R + 0.7152*G + 0.0722*B, scaled by 256 and rounded
    // (54, 183, 19) sums to 256 so white (255,255,255) correctly maps to Y=255
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        compute_y_row_simd_sse2(rgba, y_out, 54, 183, 19);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline for aarch64.
        unsafe { compute_y_row_simd_neon(rgba, y_out, 54, 183, 19) };
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    compute_y_row_scalar(rgba, y_out, 54, 183, 19);
}

fn compute_y_row_scalar(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    for (pixel, y) in rgba.chunks_exact(4).zip(y_out.iter_mut()) {
        *y = ((cr * pixel[0] as u16 + cg * pixel[1] as u16 + cb * pixel[2] as u16) >> 8) as u8;
    }
}

// x86_64: use scalar path — the compiler auto-vectorizes the simple loop with
// optimizations enabled, which outperforms the previous hand-rolled SSE2 that
// extracted individual lanes for scalar math.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn compute_y_row_simd_sse2(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    compute_y_row_scalar(rgba, y_out, cr, cg, cb);
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
unsafe fn compute_y_row_simd_neon(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    use std::arch::aarch64::*;
    let pixel_count = rgba.len() / 4;
    let simd_pixels = pixel_count / 8 * 8;

    // SAFETY: NEON is baseline for aarch64. vld4_u8 loads 8 pixels.
    unsafe {
        debug_assert!(cr <= 255, "NEON Y coefficient cr={cr} exceeds u8");
        debug_assert!(cg <= 255, "NEON Y coefficient cg={cg} exceeds u8");
        debug_assert!(cb <= 255, "NEON Y coefficient cb={cb} exceeds u8");
        let vcr = vdup_n_u8(cr as u8);
        let vcg = vdup_n_u8(cg as u8);
        let vcb = vdup_n_u8(cb as u8);

        let mut i = 0usize;
        let mut oi = 0usize;
        while oi + 8 <= simd_pixels {
            let px = vld4_u8(rgba.as_ptr().add(i));
            let mut acc = vmull_u8(px.0, vcr);
            acc = vmlal_u8(acc, px.1, vcg);
            acc = vmlal_u8(acc, px.2, vcb);
            let y = vshrn_n_u16(acc, 8);
            vst1_u8(y_out.as_mut_ptr().add(oi), y);
            i += 32;
            oi += 8;
        }
    }

    if simd_pixels < pixel_count {
        compute_y_row_scalar(
            &rgba[simd_pixels * 4..],
            &mut y_out[simd_pixels..],
            cr,
            cg,
            cb,
        );
    }
}

// =========================================================================
// BT.601 conversions
// =========================================================================

/// Convert RGBA8 buffer to YUV420p (BT.601).
///
/// Uses BT.601 fixed-point coefficients. The chroma planes (U, V) are
/// subsampled 2x2 from the top-left pixel of each block.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();
/// assert_eq!(yuv.format, PixelFormat::Yuv420p);
/// assert!(yuv.data[0] > 250);
/// ```
#[must_use = "returns a new YUV420p buffer"]
pub fn rgba_to_yuv420p(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    // Chroma loops must stay within cw*ch bounds — clamp to even pixel pairs
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 (BT.601).
///
/// Alpha is set to 255 for all pixels.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba(&yuv).unwrap();
/// assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA buffer"]
pub fn yuv420p_to_rgba(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * (h.div_ceil(2));
    let ch = h.div_ceil(2);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let cy = (y / 2).min(ch.saturating_sub(1));
        for x in 0..w {
            let cx = (x / 2).min(cw.saturating_sub(1));
            let yi = buf.data[y * w + x] as i16;
            let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
            let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
            let oi = (y * w + x) * 4;
            rgba[oi] = (yi + ((359 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 1] = (yi - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 2] = (yi + ((454 * u) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 3] = 255;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert ARGB8 buffer to NV12 (semi-planar YUV 4:2:0, BT.601).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Argb8).unwrap();
/// let nv12 = convert::argb_to_nv12(&argb).unwrap();
/// assert_eq!(nv12.format, PixelFormat::Nv12);
/// ```
#[must_use = "returns a new NV12 buffer"]
pub fn argb_to_nv12(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!(
            "argb_to_nv12: expected Argb8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let mut nv12 = vec![0u8; w * h + (w.div_ceil(2)) * (h.div_ceil(2)) * 2];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as u16;
            let g = buf.data[i + 2] as u16;
            let b = buf.data[i + 3] as u16;
            nv12[y * w + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
    }
    let uv_off = w * h;
    let nv12_cw = w.div_ceil(2);
    let nv12_ch = h.div_ceil(2);
    for y in (0..nv12_ch * 2).step_by(2) {
        for x in (0..nv12_cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as i32;
            let g = buf.data[i + 2] as i32;
            let b = buf.data[i + 3] as i32;
            let ci = (y / 2) * nv12_cw * 2 + x;
            nv12[uv_off + ci] = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            nv12[uv_off + ci + 1] =
                ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(nv12, buf.width, buf.height, PixelFormat::Nv12)
}

// =========================================================================
// BT.709 conversions (HD video — replaces tazama/tarang inline BT.709)
// =========================================================================

/// Convert RGBA8 buffer to YUV420p using BT.709 coefficients.
///
/// BT.709 is the standard for HD video. Fixed-point:
/// Y = (54*R + 183*G + 18*B) >> 8
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt709(&rgba).unwrap();
/// assert!(yuv.data[0] > 250);
/// ```
#[must_use = "returns a new YUV420p BT.709 buffer"]
pub fn rgba_to_yuv420p_bt709(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p_bt709: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row_bt709(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-29 * r - 99 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((128 * r - 116 * g - 12 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 using BT.709 coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt709(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba_bt709(&yuv).unwrap();
/// assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA BT.709 buffer"]
pub fn yuv420p_to_rgba_bt709(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba_bt709: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let cy = (y / 2).min(ch.saturating_sub(1));
        for x in 0..w {
            let cx = (x / 2).min(cw.saturating_sub(1));
            let yi = buf.data[y * w + x] as i16;
            let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
            let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
            let oi = (y * w + x) * 4;
            rgba[oi] = (yi + ((403 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 1] = (yi - ((48 * u + 120 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 2] = (yi + ((475 * u) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 3] = 255;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

// =========================================================================
// BT.2020 conversions (UHD/HDR video — wide-gamut)
// =========================================================================

/// Compute BT.2020 luminance for a row: Y = (67*R + 174*G + 15*B) >> 8
fn compute_y_row_bt2020(rgba: &[u8], y_out: &mut [u8]) {
    for (pixel, y) in rgba.chunks_exact(4).zip(y_out.iter_mut()) {
        *y = ((67 * pixel[0] as u16 + 174 * pixel[1] as u16 + 15 * pixel[2] as u16) >> 8) as u8;
    }
}

/// Convert RGBA8 buffer to YUV420p using BT.2020 coefficients.
///
/// BT.2020 is the standard for UHD/HDR video with a wider gamut than BT.709.
/// Fixed-point Y coefficients: Y = (67*R + 174*G + 15*B) >> 8
/// (derived from BT.2020 non-constant luminance: 0.2627 R + 0.6780 G + 0.0593 B)
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt2020(&rgba).unwrap();
/// assert!(yuv.data[0] > 250);
/// ```
#[must_use = "returns a new YUV420p BT.2020 buffer"]
pub fn rgba_to_yuv420p_bt2020(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p_bt2020: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row_bt2020(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    // BT.2020 Cb/Cr fixed-point coefficients (8-bit approximation)
    // Cb = (-18*R - 46*G + 64*B + 128*256) >> 8 (scaled from -0.0693*R - 0.1786*G + 0.2480*B)
    // Cr = (64*R - 58*G - 6*B + 128*256) >> 8   (scaled from 0.2480*R - 0.2252*G - 0.0228*B)
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-18 * r - 46 * g + 64 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((64 * r - 58 * g - 6 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 using BT.2020 coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt2020(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba_bt2020(&yuv).unwrap();
/// assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA BT.2020 buffer"]
pub fn yuv420p_to_rgba_bt2020(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba_bt2020: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let mut rgba = vec![0u8; w * h * 4];
    // BT.2020 inverse:
    // R = Y + 1.4746 * Cr → Y + (377 * V) >> 8
    // G = Y - 0.1646 * Cb - 0.5714 * Cr → Y - (42 * U + 146 * V) >> 8
    // B = Y + 1.8814 * Cb → Y + (481 * U) >> 8
    for y in 0..h {
        let cy = (y / 2).min(ch.saturating_sub(1));
        for x in 0..w {
            let cx = (x / 2).min(cw.saturating_sub(1));
            let yi = buf.data[y * w + x] as i16;
            let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
            let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
            let oi = (y * w + x) * 4;
            rgba[oi] = (yi + ((377 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 1] = (yi - ((42 * u + 146 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 2] = (yi + ((481 * u) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 3] = 255;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

// =========================================================================
// NV12 → RGBA (replaces tazama/tarang inline NV12 handling)
// =========================================================================

/// Convert NV12 buffer to RGBA8 (BT.601).
///
/// NV12 has a Y plane followed by interleaved UV.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255, 128, 128, 128].repeat(16), 4, 4, PixelFormat::Argb8).unwrap();
/// let nv12 = convert::argb_to_nv12(&argb).unwrap();
/// let rgba = convert::nv12_to_rgba(&nv12).unwrap();
/// assert_eq!(rgba.format, PixelFormat::Rgba8);
/// ```
#[must_use = "returns a new RGBA buffer"]
pub fn nv12_to_rgba(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Nv12 {
        return Err(RangaError::InvalidFormat(format!(
            "nv12_to_rgba: expected Nv12, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_off = w * h;
    let uv_stride = cw * 2; // interleaved UV pairs per row
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let cy = (y / 2).min(ch.saturating_sub(1));
        for x in 0..w {
            let cx = (x / 2).min(cw.saturating_sub(1));
            let yi = buf.data[y * w + x] as i16;
            let uv_idx = uv_off + cy * uv_stride + cx * 2;
            let u = buf.data[uv_idx] as i16 - 128;
            let v = buf.data[uv_idx + 1] as i16 - 128;
            let oi = (y * w + x) * 4;
            rgba[oi] = (yi + ((359 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 1] = (yi - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 2] = (yi + ((454 * u) >> 8)).clamp(0, 255) as u8;
            rgba[oi + 3] = 255;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

// =========================================================================
// Format conversions: RGB8 ↔ RGBA8, ARGB8 ↔ RGBA8, RgbaF32 ↔ Rgba8
// =========================================================================

/// Convert RGB8 to RGBA8 (add alpha=255).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgb = PixelBuffer::new(vec![255, 0, 0, 0, 255, 0], 2, 1, PixelFormat::Rgb8).unwrap();
/// let rgba = convert::rgb8_to_rgba8(&rgb).unwrap();
/// assert_eq!(rgba.data, vec![255, 0, 0, 255, 0, 255, 0, 255]);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn rgb8_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgb8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgb8_to_rgba8: expected Rgb8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        rgba[i * 4] = buf.data[i * 3];
        rgba[i * 4 + 1] = buf.data[i * 3 + 1];
        rgba[i * 4 + 2] = buf.data[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to RGB8 (strip alpha).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255, 0, 0, 128, 0, 255, 0, 64], 2, 1, PixelFormat::Rgba8).unwrap();
/// let rgb = convert::rgba8_to_rgb8(&rgba).unwrap();
/// assert_eq!(rgb.data, vec![255, 0, 0, 0, 255, 0]);
/// ```
#[must_use = "returns a new RGB8 buffer"]
pub fn rgba8_to_rgb8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_rgb8: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        rgb[i * 3] = buf.data[i * 4];
        rgb[i * 3 + 1] = buf.data[i * 4 + 1];
        rgb[i * 3 + 2] = buf.data[i * 4 + 2];
    }
    PixelBuffer::new(rgb, buf.width, buf.height, PixelFormat::Rgb8)
}

/// Convert ARGB8 to RGBA8 (channel reorder, used by aethersafta).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255, 128, 64, 32], 1, 1, PixelFormat::Argb8).unwrap();
/// let rgba = convert::argb8_to_rgba8(&argb).unwrap();
/// assert_eq!(rgba.data, vec![128, 64, 32, 255]);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn argb8_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!(
            "argb8_to_rgba8: expected Argb8, got {:?}",
            buf.format
        )));
    }
    let mut rgba = vec![0u8; buf.data.len()];
    for (src, dst) in buf.data.chunks_exact(4).zip(rgba.chunks_exact_mut(4)) {
        dst[0] = src[1];
        dst[1] = src[2];
        dst[2] = src[3];
        dst[3] = src[0];
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to ARGB8 (channel reorder).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128, 64, 32, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let argb = convert::rgba8_to_argb8(&rgba).unwrap();
/// assert_eq!(argb.data, vec![255, 128, 64, 32]);
/// ```
#[must_use = "returns a new ARGB8 buffer"]
pub fn rgba8_to_argb8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_argb8: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let mut argb = vec![0u8; buf.data.len()];
    for (src, dst) in buf.data.chunks_exact(4).zip(argb.chunks_exact_mut(4)) {
        dst[0] = src[3];
        dst[1] = src[0];
        dst[2] = src[1];
        dst[3] = src[2];
    }
    PixelBuffer::new(argb, buf.width, buf.height, PixelFormat::Argb8)
}

/// Convert RgbaF32 to RGBA8 (float → byte, clamped).
///
/// Each f32 channel is in 0.0–1.0 and scaled to 0–255.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let f32_data: Vec<u8> = [0.5f32, 0.0, 1.0, 1.0]
///     .iter()
///     .flat_map(|v| v.to_ne_bytes())
///     .collect();
/// let buf = PixelBuffer::new(f32_data, 1, 1, PixelFormat::RgbaF32).unwrap();
/// let rgba = convert::rgbaf32_to_rgba8(&buf).unwrap();
/// assert_eq!(rgba.data[0], 128);
/// assert_eq!(rgba.data[2], 255);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn rgbaf32_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::RgbaF32 {
        return Err(RangaError::InvalidFormat(format!(
            "rgbaf32_to_rgba8: expected RgbaF32, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let base = i * 16;
        for c in 0..4 {
            let bytes = [
                buf.data[base + c * 4],
                buf.data[base + c * 4 + 1],
                buf.data[base + c * 4 + 2],
                buf.data[base + c * 4 + 3],
            ];
            let v = f32::from_ne_bytes(bytes);
            rgba[i * 4 + c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to RgbaF32 (byte → float, 0.0–1.0).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let buf = PixelBuffer::new(vec![128, 0, 255, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let f32buf = convert::rgba8_to_rgbaf32(&buf).unwrap();
/// let r = f32::from_ne_bytes([f32buf.data[0], f32buf.data[1], f32buf.data[2], f32buf.data[3]]);
/// assert!((r - 128.0 / 255.0).abs() < 0.01);
/// ```
#[must_use = "returns a new RgbaF32 buffer"]
pub fn rgba8_to_rgbaf32(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_rgbaf32: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut f32data = vec![0u8; n * 16];
    for i in 0..n {
        for c in 0..4 {
            let v = buf.data[i * 4 + c] as f32 / 255.0;
            let base = i * 16 + c * 4;
            f32data[base..base + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }
    PixelBuffer::new(f32data, buf.width, buf.height, PixelFormat::RgbaF32)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_yuv_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        assert_eq!(yuv.format, PixelFormat::Yuv420p);
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn rgba_to_yuv_black() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        assert_eq!(yuv.data[0], 0);
    }

    #[test]
    fn yuv_to_rgba_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&rgba).unwrap();
        let back = yuv420p_to_rgba(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn wrong_format_rejected() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 3], 4, 4, PixelFormat::Rgb8).unwrap();
        assert!(rgba_to_yuv420p(&buf).is_err());
    }

    #[test]
    fn bt709_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&buf).unwrap();
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn bt709_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&rgba).unwrap();
        let back = yuv420p_to_rgba_bt709(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn bt709_different_from_bt601() {
        let red: Vec<u8> = [255, 0, 0, 255].repeat(16);
        let buf = PixelBuffer::new(red, 4, 4, PixelFormat::Rgba8).unwrap();
        let y601 = rgba_to_yuv420p(&buf).unwrap().data[0];
        let y709 = rgba_to_yuv420p_bt709(&buf).unwrap().data[0];
        assert_ne!(y601, y709, "BT.601 and BT.709 should differ for red");
    }

    #[test]
    fn nv12_to_rgba_roundtrip() {
        let argb_data: Vec<u8> = [255, 128, 128, 128].repeat(16);
        let argb = PixelBuffer::new(argb_data, 4, 4, PixelFormat::Argb8).unwrap();
        let nv12 = argb_to_nv12(&argb).unwrap();
        let rgba = nv12_to_rgba(&nv12).unwrap();
        assert_eq!(rgba.format, PixelFormat::Rgba8);
        assert!((rgba.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn rgb8_rgba8_roundtrip() {
        let rgb =
            PixelBuffer::new(vec![100, 150, 200, 50, 75, 25], 2, 1, PixelFormat::Rgb8).unwrap();
        let rgba = rgb8_to_rgba8(&rgb).unwrap();
        assert_eq!(rgba.data[3], 255);
        let back = rgba8_to_rgb8(&rgba).unwrap();
        assert_eq!(back.data, rgb.data);
    }

    #[test]
    fn argb8_rgba8_roundtrip() {
        let argb = PixelBuffer::new(vec![200, 100, 50, 25], 1, 1, PixelFormat::Argb8).unwrap();
        let rgba = argb8_to_rgba8(&argb).unwrap();
        assert_eq!(rgba.data, vec![100, 50, 25, 200]);
        let back = rgba8_to_argb8(&rgba).unwrap();
        assert_eq!(back.data, argb.data);
    }

    #[test]
    fn rgbaf32_rgba8_roundtrip() {
        let rgba = PixelBuffer::new(vec![128, 64, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let f32buf = rgba8_to_rgbaf32(&rgba).unwrap();
        let back = rgbaf32_to_rgba8(&f32buf).unwrap();
        for i in 0..4 {
            assert!(
                (rgba.data[i] as i16 - back.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}"
            );
        }
    }

    // Edge case: odd dimensions must not panic
    #[test]
    fn yuv420p_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 5 * 3 * 4], 5, 3, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        let _back = yuv420p_to_rgba(&yuv).unwrap();
    }

    #[test]
    fn yuv420p_bt709_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 7 * 5 * 4], 7, 5, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&buf).unwrap();
        let _back = yuv420p_to_rgba_bt709(&yuv).unwrap();
    }

    #[test]
    fn nv12_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 5 * 3 * 4], 5, 3, PixelFormat::Argb8).unwrap();
        let nv12 = argb_to_nv12(&buf).unwrap();
        let _rgba = nv12_to_rgba(&nv12).unwrap();
    }

    #[test]
    fn bt2020_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt2020(&buf).unwrap();
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn bt2020_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt2020(&rgba).unwrap();
        let back = yuv420p_to_rgba_bt2020(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn bt2020_different_from_bt709() {
        let red: Vec<u8> = [255, 0, 0, 255].repeat(16);
        let buf = PixelBuffer::new(red, 4, 4, PixelFormat::Rgba8).unwrap();
        let y709 = rgba_to_yuv420p_bt709(&buf).unwrap().data[0];
        let y2020 = rgba_to_yuv420p_bt2020(&buf).unwrap().data[0];
        assert_ne!(y709, y2020, "BT.709 and BT.2020 should differ for red");
    }

    #[test]
    fn yuv420p_1x1_no_panic() {
        let buf = PixelBuffer::new(vec![128; 4], 1, 1, PixelFormat::Rgba8).unwrap();
        let _yuv = rgba_to_yuv420p(&buf).unwrap();
    }
}
