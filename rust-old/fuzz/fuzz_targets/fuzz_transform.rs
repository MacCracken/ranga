#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::{self, Affine, ScaleFilter};

fuzz_target!(|data: &[u8]| {
    // Need: 2 bytes dimensions + at least 4 bytes pixel data + transform params
    if data.len() < 10 {
        return;
    }

    // Parse dimensions: clamp to 1-64
    let w = (data[0] as u32 % 64).max(1);
    let h = (data[1] as u32 % 64).max(1);

    let remaining = &data[2..];
    let rgba_size = (w * h * 4) as usize;
    let mut pixel_data = vec![0u8; rgba_size];
    let copy_len = remaining.len().min(rgba_size);
    pixel_data[..copy_len].copy_from_slice(&remaining[..copy_len]);

    let buf = match PixelBuffer::new(pixel_data, w, h, PixelFormat::Rgba8) {
        Ok(b) => b,
        Err(_) => return,
    };

    let params = if remaining.len() > copy_len {
        &remaining[copy_len..]
    } else {
        &[]
    };

    // Crop with fuzz-driven bounds
    if params.len() >= 4 {
        let left = params[0] as u32;
        let top = params[1] as u32;
        let right = params[2] as u32;
        let bottom = params[3] as u32;
        let _ = transform::crop(&buf, left, top, right, bottom);
    }

    // Resize with fuzz-driven new dimensions
    if params.len() >= 6 {
        let new_w = (params[4] as u32 % 64).max(1);
        let new_h = (params[5] as u32 % 64).max(1);
        let filter = if params.len() > 6 && params[6] % 3 == 0 {
            ScaleFilter::Nearest
        } else if params.len() > 6 && params[6] % 3 == 1 {
            ScaleFilter::Bilinear
        } else {
            ScaleFilter::Bicubic
        };
        let _ = transform::resize(&buf, new_w, new_h, filter);
    }

    // Flip operations
    let _ = transform::flip_horizontal(&buf);
    let _ = transform::flip_vertical(&buf);

    // Affine transform with fuzz-driven matrix values (clamped to -10..10)
    if params.len() >= 12 {
        let clamp_f64 = |b1: u8, b2: u8| -> f64 {
            let raw = i16::from_le_bytes([b1, b2]) as f64 / 3276.7; // maps to ~-10..10
            raw.clamp(-10.0, 10.0)
        };
        let affine = Affine {
            a: clamp_f64(params[0], params[1]),
            b: clamp_f64(params[2], params[3]),
            c: clamp_f64(params[4], params[5]),
            d: clamp_f64(params[6], params[7]),
            tx: clamp_f64(params[8], params[9]),
            ty: clamp_f64(params[10], params[11]),
        };
        let out_w = (w).min(32);
        let out_h = (h).min(32);
        let filter = if params.len() > 12 && params[12] % 2 == 0 {
            ScaleFilter::Nearest
        } else {
            ScaleFilter::Bilinear
        };
        let _ = transform::affine_transform(&buf, &affine, out_w, out_h, filter);
    }
});
