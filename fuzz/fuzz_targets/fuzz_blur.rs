#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fuzz_target!(|data: &[u8]| {
    // Need: 2 bytes dimensions + 1 byte radius + 2 bytes bilateral sigmas + at least 4 bytes pixels
    if data.len() < 9 {
        return;
    }

    // Parse dimensions: clamp to 1-64
    let w = (data[0] as u32 % 64).max(1);
    let h = (data[1] as u32 % 64).max(1);

    // Parse radius: clamp to 0-20
    let radius = (data[2] as u32) % 21;

    // Parse bilateral sigma values: map bytes to 0.1-50.0
    let sigma_space = 0.1 + (data[3] as f32 / 255.0) * 49.9;
    let sigma_color = 0.1 + (data[4] as f32 / 255.0) * 49.9;

    // Rest as pixel data, pad with zeros if needed
    let remaining = &data[5..];
    let rgba_size = (w * h * 4) as usize;
    let mut pixel_data = vec![0u8; rgba_size];
    let copy_len = remaining.len().min(rgba_size);
    pixel_data[..copy_len].copy_from_slice(&remaining[..copy_len]);

    let buf = match PixelBuffer::new(pixel_data, w, h, PixelFormat::Rgba8) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Gaussian blur
    let _ = filter::gaussian_blur(&buf, radius);

    // Box blur
    let _ = filter::box_blur(&buf, radius);

    // Unsharp mask
    let amount = if data.len() > 5 {
        (data[5] as f32 / 255.0) * 5.0
    } else {
        1.0
    };
    let _ = filter::unsharp_mask(&buf, radius, amount);

    // Median: use smaller max radius (0-5) since it's O(n*r^2)
    let median_radius = radius.min(5);
    let _ = filter::median(&buf, median_radius);

    // Bilateral
    let _ = filter::bilateral(&buf, radius.min(10), sigma_space, sigma_color);
});
