//! ICC profile parsing — generate sRGB v2, parse it, apply tone curves.

use ranga::icc::{IccProfile, ToneCurve, srgb_v2_profile};
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    // --- Generate and parse the built-in sRGB v2 profile ---
    let profile_bytes = srgb_v2_profile();
    println!("Generated sRGB v2 profile: {} bytes", profile_bytes.len());

    let profile = IccProfile::from_bytes(&profile_bytes).unwrap();
    println!(
        "Profile version: {}.{}",
        profile.version.0, profile.version.1
    );
    println!(
        "Color space: {:?}",
        std::str::from_utf8(&profile.color_space).unwrap_or("????")
    );

    // Print the 3x3 matrix (columns = rXYZ, gXYZ, bXYZ).
    println!("\nColor matrix (rXYZ | gXYZ | bXYZ):");
    for row in 0..3 {
        println!(
            "  [{:>9.6} {:>9.6} {:>9.6}]",
            profile.matrix[0][row], profile.matrix[1][row], profile.matrix[2][row],
        );
    }

    // --- Tone curve demonstration ---
    let tc = ToneCurve::Gamma(2.2);
    println!("\nToneCurve::Gamma(2.2) applied to sample values:");
    for &v in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        println!("  apply({v:.2}) = {:.6}", tc.apply(v));
    }

    // --- Apply profile to a small test buffer ---
    let w = 4u32;
    let h = 1u32;
    let test_colors: &[[u8; 4]] = &[
        [255, 0, 0, 255],     // red
        [0, 255, 0, 255],     // green
        [0, 0, 255, 255],     // blue
        [128, 128, 128, 255], // mid-gray
    ];

    let data: Vec<u8> = test_colors.iter().flat_map(|c| c.iter().copied()).collect();
    let buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

    println!("\nICC profile applied to test pixels (RGB -> XYZ):");
    for (i, pixel) in buf.data().chunks_exact(4).enumerate() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let (x, y, z) = profile.apply(r, g, b);
        println!(
            "  pixel[{i}]: RGB=({:>3},{:>3},{:>3}) -> XYZ=({:.4}, {:.4}, {:.4})",
            pixel[0], pixel[1], pixel[2], x, y, z,
        );
    }
}
