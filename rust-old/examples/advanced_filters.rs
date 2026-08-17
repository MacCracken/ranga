//! Advanced filters — bilateral, median, vignette, noise, threshold, invert.

use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn print_pixels(label: &str, buf: &PixelBuffer, count: usize) {
    println!("{label}:");
    for i in 0..count {
        let si = i * 4;
        println!(
            "  pixel[{i}]: R={:>3} G={:>3} B={:>3} A={:>3}",
            buf.data()[si],
            buf.data()[si + 1],
            buf.data()[si + 2],
            buf.data()[si + 3],
        );
    }
}

fn main() {
    let w = 8u32;
    let h = 8u32;
    let pixel_count = (w * h) as usize;

    // Create a test buffer with a smooth gradient.
    let mut data = Vec::with_capacity(pixel_count * 4);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 200.0 + 28.0) as u8;
            let g = ((y as f32 / h as f32) * 200.0 + 28.0) as u8;
            let b = 128u8;
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    let original = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();
    print_pixels("Original", &original, 4);

    // --- Salt-and-pepper noise ---
    let mut noisy = original.clone();
    filter::noise_salt_pepper(&mut noisy, 0.2, 42).unwrap();
    print_pixels("\nAfter noise_salt_pepper(density=0.2)", &noisy, 4);

    // --- Median filter (denoise) ---
    let denoised = filter::median(&noisy, 1).unwrap();
    print_pixels("\nAfter median(radius=1) denoise", &denoised, 4);

    // --- Bilateral filter (edge-preserving smoothing) ---
    let smoothed = filter::bilateral(&original, 2, 10.0, 30.0).unwrap();
    print_pixels(
        "\nAfter bilateral(radius=2, sigma_s=10, sigma_c=30)",
        &smoothed,
        4,
    );

    // --- Vignette ---
    let mut vignetted = original.clone();
    filter::vignette(&mut vignetted, 0.8).unwrap();
    print_pixels("\nAfter vignette(strength=0.8)", &vignetted, 4);

    // --- Threshold ---
    let mut thresholded = original.clone();
    filter::threshold(&mut thresholded, 128).unwrap();
    print_pixels("\nAfter threshold(level=128)", &thresholded, 4);

    // --- Invert ---
    let mut inverted = original.clone();
    filter::invert(&mut inverted).unwrap();
    print_pixels("\nAfter invert", &inverted, 4);

    // Show that inversion is its own inverse.
    filter::invert(&mut inverted).unwrap();
    let matches = original
        .data()
        .iter()
        .zip(inverted.data().iter())
        .all(|(a, b)| a == b);
    println!("\nDouble-invert matches original: {matches}");
}
