//! Layer compositing — composite_at, masks, transitions, gradients.
//!
//! Run with: `cargo run --example compositing`

use ranga::composite;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    // ── Background: solid blue, 8x8 ─────────────────────────────────────
    let bg_data: Vec<u8> = (0..8 * 8).flat_map(|_| [0u8, 0, 255, 255]).collect();
    let mut background = PixelBuffer::new(bg_data, 8, 8, PixelFormat::Rgba8).unwrap();
    println!(
        "Background: {}x{} solid blue",
        background.width(),
        background.height()
    );
    print_pixel("  bg[0,0]", &background, 0, 0);

    // ── Foreground: solid red, 4x4 ──────────────────────────────────────
    let fg_data: Vec<u8> = (0..4 * 4).flat_map(|_| [255u8, 0, 0, 255]).collect();
    let foreground = PixelBuffer::new(fg_data, 4, 4, PixelFormat::Rgba8).unwrap();
    println!(
        "Foreground: {}x{} solid red",
        foreground.width(),
        foreground.height()
    );

    // ── composite_at: overlay red onto blue at (2,2) with 50% opacity ───
    composite::composite_at(&foreground, &mut background, 2, 2, 0.5).unwrap();
    println!("\nAfter composite_at (red over blue at (2,2), 50% opacity):");
    print_pixel("  bg[0,0]", &background, 0, 0); // untouched blue
    print_pixel("  bg[3,3]", &background, 3, 3); // blended red+blue

    // ── Premultiply/unpremultiply alpha roundtrip ────────────────────────
    let original_data = vec![200u8, 100, 50, 128];
    let mut pm_buf = PixelBuffer::new(original_data, 1, 1, PixelFormat::Rgba8).unwrap();
    println!("\nPremultiply alpha roundtrip:");
    println!(
        "  before:        R={:>3} G={:>3} B={:>3} A={:>3}",
        pm_buf.data()[0],
        pm_buf.data()[1],
        pm_buf.data()[2],
        pm_buf.data()[3]
    );
    composite::premultiply_alpha(&mut pm_buf).unwrap();
    println!(
        "  premultiplied: R={:>3} G={:>3} B={:>3} A={:>3}",
        pm_buf.data()[0],
        pm_buf.data()[1],
        pm_buf.data()[2],
        pm_buf.data()[3]
    );
    composite::unpremultiply_alpha(&mut pm_buf).unwrap();
    println!(
        "  unpremultiplied: R={:>3} G={:>3} B={:>3} A={:>3}",
        pm_buf.data()[0],
        pm_buf.data()[1],
        pm_buf.data()[2],
        pm_buf.data()[3]
    );
    println!("  (slight rounding differences are expected)");

    // ── Apply mask with a gradient ──────────────────────────────────────
    // Create a 4x1 red buffer and a gradient mask (dark to bright).
    let red_data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 255]).collect();
    let mut masked = PixelBuffer::new(red_data, 4, 1, PixelFormat::Rgba8).unwrap();

    let mask_data: Vec<u8> = (0..4u8)
        .flat_map(|x| {
            let v = (x as f32 / 3.0 * 255.0) as u8;
            [v, v, v, 255]
        })
        .collect();
    let mask = PixelBuffer::new(mask_data, 4, 1, PixelFormat::Rgba8).unwrap();

    composite::apply_mask(&mut masked, &mask).unwrap();
    println!("\nApply mask (gradient left-to-right on red buffer):");
    for x in 0..4 {
        let i = x * 4;
        println!(
            "  pixel[{x}]: R={:>3} A={:>3}",
            masked.data()[i],
            masked.data()[i + 3]
        );
    }

    // ── Dissolve transition at midpoint ─────────────────────────────────
    let a_data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 255]).collect();
    let b_data: Vec<u8> = (0..4).flat_map(|_| [0u8, 0, 255, 255]).collect();
    let buf_a = PixelBuffer::new(a_data, 4, 1, PixelFormat::Rgba8).unwrap();
    let buf_b = PixelBuffer::new(b_data, 4, 1, PixelFormat::Rgba8).unwrap();
    let dissolved = composite::dissolve(&buf_a, &buf_b, 0.5).unwrap();
    println!("\nDissolve (red -> blue at 50%):");
    print_pixel("  pixel[0]", &dissolved, 0, 0);
    println!("  (R and B should both be ~128)");

    // ── Gradient linear fill ────────────────────────────────────────────
    let mut grad_buf = PixelBuffer::zeroed(8, 1, PixelFormat::Rgba8);
    composite::gradient_linear(
        &mut grad_buf,
        [255, 0, 0, 255], // red on the left
        [0, 0, 255, 255], // blue on the right
    )
    .unwrap();
    println!("\nGradient linear (red -> blue, 8x1):");
    for x in 0..8 {
        let i = x * 4;
        let d = grad_buf.data();
        println!(
            "  pixel[{x}]: R={:>3} G={:>3} B={:>3}",
            d[i],
            d[i + 1],
            d[i + 2]
        );
    }
}

/// Print RGBA values of a single pixel at (x, y).
fn print_pixel(label: &str, buf: &PixelBuffer, x: usize, y: usize) {
    let i = (y * buf.width() as usize + x) * 4;
    let d = buf.data();
    println!(
        "{label}: R={:>3} G={:>3} B={:>3} A={:>3}",
        d[i],
        d[i + 1],
        d[i + 2],
        d[i + 3]
    );
}
