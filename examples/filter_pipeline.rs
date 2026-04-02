use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    let w = 4u32;
    let h = 4u32;

    // Create an RGBA8 buffer with varied pixel values
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) as u8 {
        let r = i.wrapping_mul(17);
        let g = i.wrapping_mul(31);
        let b = i.wrapping_mul(53);
        data.extend_from_slice(&[r, g, b, 255]);
    }

    let mut buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

    // Print before
    println!("Before filters:");
    for (i, pixel) in buf.data().chunks_exact(4).enumerate().take(4) {
        println!(
            "  pixel[{i}]: R={:>3} G={:>3} B={:>3} A={:>3}",
            pixel[0], pixel[1], pixel[2], pixel[3]
        );
    }

    // Apply brightness (+0.1), then contrast (1.5x), then grayscale
    filter::brightness(&mut buf, 0.1).unwrap();
    filter::contrast(&mut buf, 1.5).unwrap();
    filter::grayscale(&mut buf).unwrap();

    // Print after
    println!("\nAfter brightness(+0.1) -> contrast(1.5) -> grayscale:");
    for (i, pixel) in buf.data().chunks_exact(4).enumerate().take(4) {
        println!(
            "  pixel[{i}]: R={:>3} G={:>3} B={:>3} A={:>3}",
            pixel[0], pixel[1], pixel[2], pixel[3]
        );
    }

    // Verify grayscale: R == G == B for every pixel
    let all_gray = buf
        .data()
        .chunks_exact(4)
        .all(|p| p[0] == p[1] && p[1] == p[2]);
    println!("\nAll pixels are grayscale: {all_gray}");
}
