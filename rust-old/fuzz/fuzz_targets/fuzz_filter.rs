#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fuzz_target!(|data: &[u8]| {
    // Need: 2 bytes for dimensions + at least 4 bytes for one pixel + 8 bytes for params
    if data.len() < 14 {
        return;
    }

    // Small even dimensions
    let w = ((data[0] as u32 & 0x1E).max(2)) & !1; // even, 2..30
    let h = ((data[1] as u32 & 0x1E).max(2)) & !1; // even, 2..30
    let rgba_size = (w * h * 4) as usize;

    let remaining = &data[2..];
    if remaining.len() < rgba_size + 8 {
        return;
    }

    let pixel_data = remaining[..rgba_size].to_vec();
    let params = &remaining[rgba_size..];

    // Extract filter parameters from the remaining fuzz bytes
    let brightness_offset = if params.len() >= 4 {
        f32::from_le_bytes([params[0], params[1], params[2], params[3]])
    } else {
        0.0
    };

    let contrast_factor = if params.len() >= 8 {
        f32::from_le_bytes([params[4], params[5], params[6], params[7]])
    } else {
        1.0
    };

    // Clamp to avoid extreme floats (NaN, Inf) that aren't meaningful inputs
    let brightness_offset = if brightness_offset.is_finite() {
        brightness_offset.clamp(-10.0, 10.0)
    } else {
        0.0
    };
    let contrast_factor = if contrast_factor.is_finite() {
        contrast_factor.clamp(-10.0, 10.0)
    } else {
        1.0
    };

    // brightness
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::brightness(&mut buf, brightness_offset);
    }

    // contrast
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::contrast(&mut buf, contrast_factor);
    }

    // saturation
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::saturation(&mut buf, contrast_factor);
    }

    // grayscale
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::grayscale(&mut buf);
    }

    // invert
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::invert(&mut buf);
    }

    // levels
    if let Ok(mut buf) = PixelBuffer::new(pixel_data.clone(), w, h, PixelFormat::Rgba8) {
        let _ = filter::levels(&mut buf, 0.0, 1.0, contrast_factor.abs().max(0.01));
    }

    // curves with an identity LUT
    if let Ok(mut buf) = PixelBuffer::new(pixel_data, w, h, PixelFormat::Rgba8) {
        let mut lut = [0u8; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            // Use a fuzz byte to offset the LUT if available
            let offset = if params.len() > 8 { params[8] } else { 0 };
            *entry = (i as u8).wrapping_add(offset);
        }
        let _ = filter::curves(&mut buf, &lut);
    }
});
