//! Transform operations — crop, resize, flip, affine, perspective.
//!
//! Run with: `cargo run --example transforms`

use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::{self, Affine, Perspective, ScaleFilter};
use std::f64::consts::PI;

fn main() {
    let w = 8u32;
    let h = 8u32;
    let pixel_count = (w * h) as usize;

    // Create an 8x8 buffer with a gradient pattern: R increases left-to-right,
    // G increases top-to-bottom, B fixed at 128.
    let mut data = Vec::with_capacity(pixel_count * 4);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / (w - 1) as f32) * 255.0) as u8;
            let g = ((y as f32 / (h - 1) as f32) * 255.0) as u8;
            let b = 128u8;
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    let buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();
    println!("Source buffer: {}x{}", buf.width(), buf.height());
    print_corner_pixels("  source", &buf);

    // ── Crop ──────────────────────────────────────────────────────────────
    // Extract a 4x4 region from the center.
    let cropped = transform::crop(&buf, 2, 2, 6, 6).unwrap();
    println!(
        "\nCrop (2,2)-(6,6): {}x{}",
        cropped.width(),
        cropped.height()
    );
    print_corner_pixels("  cropped", &cropped);

    // ── Resize with each ScaleFilter ─────────────────────────────────────
    for filter in [
        ScaleFilter::Nearest,
        ScaleFilter::Bilinear,
        ScaleFilter::Bicubic,
    ] {
        let resized = transform::resize(&buf, 4, 4, filter).unwrap();
        println!(
            "\nResize 8x8 -> 4x4 ({filter:?}): {}x{}",
            resized.width(),
            resized.height()
        );
        print_corner_pixels("  resized", &resized);
    }

    // ── Flip horizontal ──────────────────────────────────────────────────
    let flipped_h = transform::flip_horizontal(&buf).unwrap();
    println!(
        "\nFlip horizontal: {}x{}",
        flipped_h.width(),
        flipped_h.height()
    );
    println!(
        "  top-left was R={}, now R={} (swapped with top-right)",
        buf.data()[0],
        flipped_h.data()[0]
    );

    // ── Flip vertical ────────────────────────────────────────────────────
    let flipped_v = transform::flip_vertical(&buf).unwrap();
    println!(
        "\nFlip vertical: {}x{}",
        flipped_v.width(),
        flipped_v.height()
    );
    println!(
        "  top-left was G={}, now G={} (swapped with bottom-left)",
        buf.data()[1],
        flipped_v.data()[1]
    );

    // ── Affine rotate 45° ────────────────────────────────────────────────
    // Rotate around the center of the image.
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let rotate_45 = Affine::translate(cx, cy)
        .then(&Affine::rotate(PI / 4.0))
        .then(&Affine::translate(-cx, -cy));
    let rotated =
        transform::affine_transform(&buf, &rotate_45, w, h, ScaleFilter::Bilinear).unwrap();
    println!(
        "\nAffine rotate 45°: {}x{}",
        rotated.width(),
        rotated.height()
    );
    print_corner_pixels("  rotated", &rotated);

    // ── Perspective transform ────────────────────────────────────────────
    // Mild keystone correction: pinch the top edge inward.
    let src_corners = [
        (0.0, 0.0),
        (w as f64, 0.0),
        (w as f64, h as f64),
        (0.0, h as f64),
    ];
    let dst_corners = [
        (1.0, 0.0),
        ((w - 1) as f64, 0.0),
        (w as f64, h as f64),
        (0.0, h as f64),
    ];
    let persp =
        Perspective::from_quad(src_corners, dst_corners).expect("perspective should be solvable");
    let perspected =
        transform::perspective_transform(&buf, &persp, w, h, ScaleFilter::Bilinear).unwrap();
    println!(
        "\nPerspective (keystone): {}x{}",
        perspected.width(),
        perspected.height()
    );
    print_corner_pixels("  perspected", &perspected);
}

/// Print the four corner pixels of a buffer for quick inspection.
fn print_corner_pixels(label: &str, buf: &PixelBuffer) {
    let w = buf.width() as usize;
    let h = buf.height() as usize;
    let corners = [
        (0, 0, "top-left"),
        (w - 1, 0, "top-right"),
        (0, h - 1, "bottom-left"),
        (w - 1, h - 1, "bottom-right"),
    ];
    for (x, y, name) in corners {
        let i = (y * w + x) * 4;
        let d = buf.data();
        println!(
            "{label} {name}: R={:>3} G={:>3} B={:>3} A={:>3}",
            d[i],
            d[i + 1],
            d[i + 2],
            d[i + 3]
        );
    }
}
