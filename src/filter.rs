//! CPU image filters — brightness, contrast, saturation, levels, curves.

use crate::pixel::{PixelBuffer, PixelFormat};
use crate::RangaError;

/// Adjust brightness of an RGBA8 buffer in-place.
/// `offset` is in -1.0 to 1.0 range (maps to -255 to +255).
pub fn brightness(buf: &mut PixelBuffer, offset: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let shift = (offset * 255.0) as i16;
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = (pixel[0] as i16 + shift).clamp(0, 255) as u8;
        pixel[1] = (pixel[1] as i16 + shift).clamp(0, 255) as u8;
        pixel[2] = (pixel[2] as i16 + shift).clamp(0, 255) as u8;
    }
    Ok(())
}

/// Adjust contrast of an RGBA8 buffer in-place.
/// `factor` of 1.0 is unchanged; >1.0 increases, <1.0 decreases.
pub fn contrast(buf: &mut PixelBuffer, factor: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = ((((pixel[0] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
        pixel[1] = ((((pixel[1] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
        pixel[2] = ((((pixel[2] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
    }
    Ok(())
}

/// Adjust saturation of an RGBA8 buffer in-place.
/// `factor` of 1.0 is unchanged; 0.0 is grayscale; >1.0 increases saturation.
pub fn saturation(buf: &mut PixelBuffer, factor: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        pixel[0] = (gray + factor * (r - gray)).clamp(0.0, 255.0) as u8;
        pixel[1] = (gray + factor * (g - gray)).clamp(0.0, 255.0) as u8;
        pixel[2] = (gray + factor * (b - gray)).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Apply levels adjustment (black point, white point, gamma).
/// All values in 0.0–1.0 range. Gamma of 1.0 is linear.
pub fn levels(
    buf: &mut PixelBuffer,
    black: f32,
    white: f32,
    gamma: f32,
) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let range = (white - black).max(1e-6);
    // Build LUT for speed
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = i as f32 / 255.0;
        let mapped = ((v - black) / range).clamp(0.0, 1.0);
        let corrected = mapped.powf(1.0 / gamma);
        *entry = (corrected * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    Ok(())
}

/// Apply a curves adjustment using a 256-entry lookup table.
pub fn curves(buf: &mut PixelBuffer, lut: &[u8; 256]) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    Ok(())
}

/// Convert an RGBA8 buffer to grayscale in-place (BT.601 luminance).
pub fn grayscale(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let gray = ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8;
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }
    Ok(())
}

/// Invert all color channels in-place.
pub fn invert(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = 255 - pixel[0];
        pixel[1] = 255 - pixel[1];
        pixel[2] = 255 - pixel[2];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buf() -> PixelBuffer {
        PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap()
    }

    #[test]
    fn brightness_positive() {
        let mut buf = test_buf();
        brightness(&mut buf, 0.5).unwrap();
        assert!(buf.data[0] > 200);
    }

    #[test]
    fn brightness_negative() {
        let mut buf = test_buf();
        brightness(&mut buf, -0.5).unwrap();
        assert!(buf.data[0] < 10);
    }

    #[test]
    fn contrast_increase() {
        let mut buf = test_buf();
        contrast(&mut buf, 2.0).unwrap();
        // 128 is center → stays at 128
        assert_eq!(buf.data[0], 128);
    }

    #[test]
    fn saturation_zero_is_gray() {
        let mut buf = PixelBuffer::new(
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            2, 1, PixelFormat::Rgba8,
        ).unwrap();
        saturation(&mut buf, 0.0).unwrap();
        // All channels should be equal (gray)
        assert_eq!(buf.data[0], buf.data[1]);
        assert_eq!(buf.data[1], buf.data[2]);
    }

    #[test]
    fn grayscale_makes_uniform() {
        let mut buf = PixelBuffer::new(
            vec![200, 100, 50, 255],
            1, 1, PixelFormat::Rgba8,
        ).unwrap();
        grayscale(&mut buf).unwrap();
        assert_eq!(buf.data[0], buf.data[1]);
        assert_eq!(buf.data[1], buf.data[2]);
        assert_eq!(buf.data[3], 255); // alpha unchanged
    }

    #[test]
    fn invert_roundtrip() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        invert(&mut buf).unwrap();
        invert(&mut buf).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn levels_identity() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        levels(&mut buf, 0.0, 1.0, 1.0).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn curves_identity() {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        let mut buf = test_buf();
        let original = buf.data.clone();
        curves(&mut buf, &lut).unwrap();
        assert_eq!(buf.data, original);
    }
}
