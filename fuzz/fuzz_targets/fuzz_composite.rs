#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::composite;
use ranga::pixel::{PixelBuffer, PixelFormat};

fuzz_target!(|data: &[u8]| {
    // Need: 4 bytes for dimensions (w1,h1,w2,h2) + 1 byte progress + pixel data
    if data.len() < 9 {
        return;
    }

    // Parse dimensions for two buffers, clamped to 1-32
    let w1 = (data[0] as u32 % 32).max(1);
    let h1 = (data[1] as u32 % 32).max(1);
    let w2 = (data[2] as u32 % 32).max(1);
    let h2 = (data[3] as u32 % 32).max(1);

    let progress = data[4] as f32 / 255.0;
    let remaining = &data[5..];

    let rgba_size1 = (w1 * h1 * 4) as usize;
    let rgba_size2 = (w2 * h2 * 4) as usize;

    // Build buffer 1
    let mut pixel_data1 = vec![0u8; rgba_size1];
    let copy1 = remaining.len().min(rgba_size1);
    pixel_data1[..copy1].copy_from_slice(&remaining[..copy1]);

    // Build buffer 2 from remaining bytes after buffer 1
    let rest = if remaining.len() > copy1 {
        &remaining[copy1..]
    } else {
        &[]
    };
    let mut pixel_data2 = vec![0u8; rgba_size2];
    let copy2 = rest.len().min(rgba_size2);
    if copy2 > 0 {
        pixel_data2[..copy2].copy_from_slice(&rest[..copy2]);
    }

    let buf1 = match PixelBuffer::new(pixel_data1, w1, h1, PixelFormat::Rgba8) {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut buf2 = match PixelBuffer::new(pixel_data2, w2, h2, PixelFormat::Rgba8) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Dissolve: requires same size, so only try if dimensions match
    if w1 == w2 && h1 == h2 {
        let _ = composite::dissolve(&buf1, &buf2, progress);
    }

    // composite_at with fuzz-driven offsets
    let x_off = if data.len() > 5 {
        data[5] as i32 - 128
    } else {
        0
    };
    let y_off = if data.len() > 6 {
        data[6] as i32 - 128
    } else {
        0
    };
    let _ = composite::composite_at(&buf1, &mut buf2, x_off, y_off, progress);

    // apply_mask: requires same size
    if w1 == w2 && h1 == h2 {
        let mut buf1_clone = buf1.clone();
        let _ = composite::apply_mask(&mut buf1_clone, &buf2);
    }

    // premultiply→unpremultiply roundtrip
    {
        let mut rt_buf = buf1.clone();
        if composite::premultiply_alpha(&mut rt_buf).is_ok() {
            let _ = composite::unpremultiply_alpha(&mut rt_buf);
        }
    }
});
