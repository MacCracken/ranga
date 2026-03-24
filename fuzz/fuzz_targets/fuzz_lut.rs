#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::filter::{self, Lut3d};
use ranga::pixel::{PixelBuffer, PixelFormat};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Strategy 1: Feed raw bytes as potential .cube file content
    if let Ok(text) = std::str::from_utf8(data)
        && let Ok(lut) = Lut3d::from_cube(text)
    {
        // If it parses, apply it to a small test buffer
        let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        // Fill with some non-zero data
        for pixel in buf.data.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 200;
            pixel[3] = 255;
        }
        let _ = filter::apply_lut3d(&mut buf, &lut);
    }

    // Strategy 2: Build a valid cube string with fuzz-driven float values
    if data.len() >= 3 {
        // Use first byte for LUT size (2-5 to keep it small)
        let size = (data[0] as usize % 4) + 2;
        let total_entries = size * size * size;

        let mut cube_str = format!("LUT_3D_SIZE {size}\n");
        let float_bytes = &data[1..];

        for i in 0..total_entries {
            let base = (i * 3) % float_bytes.len().max(1);
            let r = (float_bytes[base % float_bytes.len()] as f32) / 255.0;
            let g = (float_bytes[(base + 1) % float_bytes.len()] as f32) / 255.0;
            let b = (float_bytes[(base + 2) % float_bytes.len()] as f32) / 255.0;
            cube_str.push_str(&format!("{r} {g} {b}\n"));
        }

        if let Ok(lut) = Lut3d::from_cube(&cube_str) {
            let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
            for (i, pixel) in buf.data.chunks_exact_mut(4).enumerate() {
                let idx = i % float_bytes.len().max(1);
                pixel[0] = float_bytes[idx];
                pixel[1] = float_bytes[(idx + 1) % float_bytes.len()];
                pixel[2] = float_bytes[(idx + 2) % float_bytes.len()];
                pixel[3] = 255;
            }
            let _ = filter::apply_lut3d(&mut buf, &lut);
        }
    }
});
