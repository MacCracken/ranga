//! Pixel format conversion — RGB↔YUV, ARGB↔NV12, etc.
//!
//! Uses fixed-point BT.601 coefficients for integer-only fast paths
//! and floating-point BT.709 for HDR/linear workflows.

use crate::pixel::{PixelBuffer, PixelFormat};
use crate::RangaError;

/// Convert RGBA8 buffer to YUV420p.
pub fn rgba_to_yuv420p(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }

    let w = buf.width as usize;
    let h = buf.height as usize;
    let chroma_w = w / 2;
    let chroma_h = h / 2;
    let mut yuv = vec![0u8; w * h + 2 * chroma_w * chroma_h];

    // Y plane — BT.601 fixed-point
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as u16;
            let g = buf.data[i + 1] as u16;
            let b = buf.data[i + 2] as u16;
            // Y = (77*R + 150*G + 29*B) >> 8
            yuv[y * w + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
    }

    // U and V planes (subsampled 2x2)
    let u_offset = w * h;
    let v_offset = u_offset + chroma_w * chroma_h;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let u = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            let v = ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            let ci = (y / 2) * chroma_w + (x / 2);
            yuv[u_offset + ci] = u;
            yuv[v_offset + ci] = v;
        }
    }

    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8.
pub fn yuv420p_to_rgba(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }

    let w = buf.width as usize;
    let h = buf.height as usize;
    let chroma_w = w / 2;
    let u_offset = w * h;
    let v_offset = u_offset + chroma_w * (h / 2);

    let mut rgba = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let yi = buf.data[y * w + x] as i16;
            let u = buf.data[u_offset + (y / 2) * chroma_w + (x / 2)] as i16 - 128;
            let v = buf.data[v_offset + (y / 2) * chroma_w + (x / 2)] as i16 - 128;

            let r = (yi + ((359 * v) >> 8)).clamp(0, 255) as u8;
            let g = (yi - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
            let b = (yi + ((454 * u) >> 8)).clamp(0, 255) as u8;

            let oi = (y * w + x) * 4;
            rgba[oi] = r;
            rgba[oi + 1] = g;
            rgba[oi + 2] = b;
            rgba[oi + 3] = 255;
        }
    }

    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert ARGB8 buffer to NV12 (semi-planar YUV 4:2:0).
pub fn argb_to_nv12(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }

    let w = buf.width as usize;
    let h = buf.height as usize;
    let mut nv12 = vec![0u8; w * h + (w / 2) * (h / 2) * 2];

    // Y plane
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as u16;
            let g = buf.data[i + 2] as u16;
            let b = buf.data[i + 3] as u16;
            nv12[y * w + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
    }

    // UV plane (interleaved)
    let uv_offset = w * h;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as i32;
            let g = buf.data[i + 2] as i32;
            let b = buf.data[i + 3] as i32;
            let u = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            let v = ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            let ci = (y / 2) * w + x;
            nv12[uv_offset + ci] = u;
            nv12[uv_offset + ci + 1] = v;
        }
    }

    PixelBuffer::new(nv12, buf.width, buf.height, PixelFormat::Nv12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_yuv_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        assert_eq!(yuv.format, PixelFormat::Yuv420p);
        // White → Y ≈ 255
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn rgba_to_yuv_black() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        // Black → Y = 0
        assert_eq!(yuv.data[0], 0);
    }

    #[test]
    fn yuv_to_rgba_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&rgba).unwrap();
        let back = yuv420p_to_rgba(&yuv).unwrap();
        assert_eq!(back.format, PixelFormat::Rgba8);
        assert_eq!(back.data.len(), 8 * 8 * 4);
        // Should be close to original (lossy conversion)
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn wrong_format_rejected() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 3], 4, 4, PixelFormat::Rgb8).unwrap();
        assert!(rgba_to_yuv420p(&buf).is_err());
    }
}
