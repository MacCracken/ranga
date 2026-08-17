use ranga::convert::{rgba_to_yuv420p, yuv420p_to_rgba};
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    let w = 8u32;
    let h = 8u32;
    let pixel_count = (w * h) as usize;

    // Create an RGBA8 buffer with a gradient pattern
    let mut data = Vec::with_capacity(pixel_count * 4);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            let b = 128u8;
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }

    let original = data.clone();
    let buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

    // Convert to YUV420p
    let yuv = rgba_to_yuv420p(&buf).unwrap();
    println!(
        "Converted {}x{} RGBA8 ({} bytes) -> YUV420p ({} bytes)",
        w,
        h,
        original.len(),
        yuv.data().len()
    );

    // Convert back to RGBA8
    let back = yuv420p_to_rgba(&yuv).unwrap();

    // Find the maximum per-channel difference to show the lossy nature
    let mut max_diff: u16 = 0;
    let mut total_diff: u64 = 0;
    let mut samples: u64 = 0;

    for (orig_pixel, round_pixel) in original.chunks_exact(4).zip(back.data().chunks_exact(4)) {
        for c in 0..3 {
            let d = (orig_pixel[c] as i16 - round_pixel[c] as i16).unsigned_abs();
            if d > max_diff {
                max_diff = d;
            }
            total_diff += d as u64;
            samples += 1;
        }
    }

    let avg_diff = total_diff as f64 / samples as f64;
    println!("Max per-channel difference: {max_diff}");
    println!("Average per-channel difference: {avg_diff:.2}");
    println!("(Non-zero difference is expected due to YUV chroma subsampling and rounding)");
}
