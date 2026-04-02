//! Histogram analysis — luminance histogram, RGB histograms, equalization, auto-levels.

use ranga::histogram;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    let w = 16u32;
    let h = 16u32;
    let pixel_count = (w * h) as usize;

    // Create a buffer with a gradient pattern (dark to bright).
    let mut data = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let v = ((i as f32 / pixel_count as f32) * 255.0) as u8;
        data.extend_from_slice(&[v, v, v, 255]);
    }
    let buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

    // --- Luminance histogram (256 bins) ---
    let lum_hist = histogram::luminance_histogram(&buf, 256).unwrap();

    // Find peak bin and its value.
    let (peak_bin, peak_val) = lum_hist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("Luminance histogram (256 bins):");
    println!("  peak bin: {peak_bin} ({peak_val:.4})");
    let sum: f64 = lum_hist.iter().sum();
    println!("  sum: {sum:.6} (should be ~1.0)");

    // Print distribution quartiles.
    let mut cumulative = 0.0;
    let mut q25 = 0;
    let mut q50 = 0;
    let mut q75 = 0;
    for (i, &v) in lum_hist.iter().enumerate() {
        cumulative += v;
        if cumulative >= 0.25 && q25 == 0 {
            q25 = i;
        }
        if cumulative >= 0.50 && q50 == 0 {
            q50 = i;
        }
        if cumulative >= 0.75 && q75 == 0 {
            q75 = i;
        }
    }
    println!("  quartiles: Q25={q25} Q50={q50} Q75={q75}");

    // --- RGB histograms ---
    let [r_hist, g_hist, b_hist] = histogram::rgb_histograms(&buf).unwrap();
    println!("\nRGB histograms (256 bins each):");
    println!(
        "  R non-zero bins: {}",
        r_hist.iter().filter(|&&v| v > 0.0).count()
    );
    println!(
        "  G non-zero bins: {}",
        g_hist.iter().filter(|&&v| v > 0.0).count()
    );
    println!(
        "  B non-zero bins: {}",
        b_hist.iter().filter(|&&v| v > 0.0).count()
    );

    // --- Equalize a copy ---
    let mut equalized = buf.clone();
    histogram::equalize(&mut equalized).unwrap();

    println!("\nBefore vs. after equalization (first 4 pixels):");
    for i in 0..4 {
        let si = i * 4;
        println!(
            "  pixel[{i}]: original=({:>3},{:>3},{:>3})  equalized=({:>3},{:>3},{:>3})",
            buf.data()[si],
            buf.data()[si + 1],
            buf.data()[si + 2],
            equalized.data()[si],
            equalized.data()[si + 1],
            equalized.data()[si + 2],
        );
    }

    // --- Auto-levels on a copy ---
    let mut leveled = buf.clone();
    histogram::auto_levels(&mut leveled).unwrap();

    println!("\nBefore vs. after auto-levels (first 4 pixels):");
    for i in 0..4 {
        let si = i * 4;
        println!(
            "  pixel[{i}]: original=({:>3},{:>3},{:>3})  leveled=({:>3},{:>3},{:>3})",
            buf.data()[si],
            buf.data()[si + 1],
            buf.data()[si + 2],
            leveled.data()[si],
            leveled.data()[si + 1],
            leveled.data()[si + 2],
        );
    }

    // --- Chi-squared distance between original and equalized histograms ---
    let eq_hist = histogram::luminance_histogram(&equalized, 256).unwrap();
    let distance = histogram::chi_squared(&lum_hist, &eq_hist).unwrap();
    println!("\nChi-squared distance (original vs. equalized): {distance:.6}");
    if distance > 0.0 {
        println!("  Histograms differ — equalization redistributed intensities.");
    } else {
        println!("  Histograms are identical.");
    }
}
