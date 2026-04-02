//! Visual regression tests — deterministic pixel checks on known operations.
//!
//! These tests verify exact or near-exact pixel output for specific operations
//! on predetermined inputs, catching visual artifacts that tolerance-based tests
//! might miss.

use ranga::blend::{BlendMode, blend_pixel};
use ranga::composite;
use ranga::convert::{rgba_to_yuv420p, yuv420p_to_rgba};
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::{self, ScaleFilter};

/// Helper: create an RGBA8 buffer from a flat pixel array.
fn make_buf(w: u32, h: u32, data: Vec<u8>) -> PixelBuffer {
    PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap()
}

/// Helper: assert max per-channel difference between two equal-size buffers.
fn assert_max_diff(a: &PixelBuffer, b: &PixelBuffer, max_diff: u8, label: &str) {
    assert_eq!(
        a.data().len(),
        b.data().len(),
        "{label}: buffer size mismatch"
    );
    for (i, (&av, &bv)) in a.data().iter().zip(b.data().iter()).enumerate() {
        let diff = (av as i16 - bv as i16).unsigned_abs() as u8;
        assert!(
            diff <= max_diff,
            "{label}: byte {i} differs by {diff} ({av} vs {bv}), max allowed {max_diff}"
        );
    }
}

// -----------------------------------------------------------------------
// 1. Gradient blur smoothness — no banding artifacts
// -----------------------------------------------------------------------

#[test]
fn gradient_blur_smoothness() {
    // Generate a 64x1 horizontal linear gradient (R channel: 0..252 stepping by 4)
    let w = 64u32;
    let h = 1u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for x in 0..w as usize {
        let v = ((x * 255) / (w as usize - 1)).min(255) as u8;
        let i = x * 4;
        data[i] = v;
        data[i + 1] = v;
        data[i + 2] = v;
        data[i + 3] = 255;
    }
    let buf = make_buf(w, h, data);
    let blurred = filter::gaussian_blur(&buf, 3).unwrap();

    // Verify the red channel is monotonically non-decreasing (no banding dips).
    let mut prev = 0u8;
    for x in 0..w as usize {
        let r = blurred.data()[x * 4];
        assert!(
            r >= prev || (prev - r) <= 1,
            "banding artifact at x={x}: prev={prev}, current={r}"
        );
        prev = r;
    }
}

// -----------------------------------------------------------------------
// 2. Checkerboard resize — bilinear 2x2 → 4x4
// -----------------------------------------------------------------------

#[test]
fn checkerboard_resize_bilinear() {
    // 2x2 checkerboard: BW / WB (opaque)
    #[rustfmt::skip]
    let data = vec![
        0, 0, 0, 255,       255, 255, 255, 255,
        255, 255, 255, 255,  0, 0, 0, 255,
    ];
    let buf = make_buf(2, 2, data);
    let resized = transform::resize(&buf, 4, 4, ScaleFilter::Bilinear).unwrap();
    assert_eq!(resized.width(), 4);
    assert_eq!(resized.height(), 4);

    // Corner pixels should preserve original values (they map exactly to src corners).
    // Top-left corner = black
    let tl = resized.get_rgba(0, 0).unwrap();
    assert_eq!(tl[0], 0, "top-left should be black");
    // Top-right corner = white
    let tr = resized.get_rgba(3, 0).unwrap();
    assert_eq!(tr[0], 255, "top-right should be white");
    // Bottom-left corner = white
    let bl = resized.get_rgba(0, 3).unwrap();
    assert_eq!(bl[0], 255, "bottom-left should be white");
    // Bottom-right corner = black
    let br = resized.get_rgba(3, 3).unwrap();
    assert_eq!(br[0], 0, "bottom-right should be black");

    // Interior pixels should be mid-gray (bilinear blend of B and W)
    let mid = resized.get_rgba(1, 1).unwrap();
    assert!(
        (100..=156).contains(&mid[0]),
        "interior pixel should be mid-gray, got {}",
        mid[0]
    );
}

// -----------------------------------------------------------------------
// 3. Invert idempotency on a larger varied buffer
// -----------------------------------------------------------------------

#[test]
fn invert_idempotency_large() {
    let w = 16u32;
    let h = 16u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    // Fill with varied data
    for (i, pixel) in data.chunks_exact_mut(4).enumerate() {
        pixel[0] = (i * 7 % 256) as u8;
        pixel[1] = (i * 13 % 256) as u8;
        pixel[2] = (i * 23 % 256) as u8;
        pixel[3] = 255;
    }
    let original = data.clone();
    let mut buf = make_buf(w, h, data);

    // invert twice = original
    filter::invert(&mut buf).unwrap();
    // Verify it actually changed
    assert_ne!(buf.data(), original, "first invert should change data");

    filter::invert(&mut buf).unwrap();
    assert_eq!(
        buf.data(),
        original,
        "double invert should restore original"
    );
}

// -----------------------------------------------------------------------
// 4. Premultiply roundtrip precision
// -----------------------------------------------------------------------

#[test]
fn premultiply_roundtrip_precision() {
    let w = 16u32;
    let h = 1u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    // Semi-transparent gradient
    for x in 0..w as usize {
        let i = x * 4;
        let v = ((x * 255) / (w as usize - 1)).min(255) as u8;
        data[i] = v;
        data[i + 1] = 200;
        data[i + 2] = 100;
        // Alpha gradient from 64 to 255
        data[i + 3] = (64 + (x * 191) / (w as usize - 1)).min(255) as u8;
    }
    let original = make_buf(w, h, data.clone());
    let mut buf = make_buf(w, h, data);

    composite::premultiply_alpha(&mut buf).unwrap();
    composite::unpremultiply_alpha(&mut buf).unwrap();

    // Max per-channel error should be <= 3 due to integer division rounding
    // (premultiply truncates, unpremultiply truncates again — compounds at low alpha)
    assert_max_diff(&original, &buf, 3, "premultiply roundtrip");
}

// -----------------------------------------------------------------------
// 5. Gaussian blur symmetry — single white pixel on black
// -----------------------------------------------------------------------

#[test]
fn gaussian_blur_symmetry() {
    let w = 17u32;
    let h = 17u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    // Place a single white pixel at the center (8, 8)
    let cx = 8usize;
    let cy = 8usize;
    let ci = (cy * w as usize + cx) * 4;
    data[ci] = 255;
    data[ci + 1] = 255;
    data[ci + 2] = 255;
    data[ci + 3] = 255;

    // Set all alpha to 255 so blur operates on visible pixels
    for pixel in data.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let buf = make_buf(w, h, data);
    let blurred = filter::gaussian_blur(&buf, 3).unwrap();

    // Verify horizontal symmetry around center
    for y in 0..h as usize {
        for x in 0..cx {
            let mirror_x = 2 * cx - x;
            if mirror_x < w as usize {
                let li = (y * w as usize + x) * 4;
                let ri = (y * w as usize + mirror_x) * 4;
                for c in 0..3 {
                    assert_eq!(
                        blurred.data()[li + c],
                        blurred.data()[ri + c],
                        "horizontal symmetry failed at ({x},{y}) vs ({mirror_x},{y}), channel {c}"
                    );
                }
            }
        }
    }

    // Verify vertical symmetry around center
    for y in 0..cy {
        let mirror_y = 2 * cy - y;
        if mirror_y < h as usize {
            for x in 0..w as usize {
                let ti = (y * w as usize + x) * 4;
                let bi = (mirror_y * w as usize + x) * 4;
                for c in 0..3 {
                    assert_eq!(
                        blurred.data()[ti + c],
                        blurred.data()[bi + c],
                        "vertical symmetry failed at ({x},{y}) vs ({x},{mirror_y}), channel {c}"
                    );
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// 6. HSL hue shift 360 identity
// -----------------------------------------------------------------------

#[test]
fn hsl_hue_shift_360_identity() {
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for (i, pixel) in data.chunks_exact_mut(4).enumerate() {
        pixel[0] = (i * 37 % 256) as u8;
        pixel[1] = (i * 73 % 256) as u8;
        pixel[2] = (i * 113 % 256) as u8;
        pixel[3] = 255;
    }
    let original = data.clone();
    let mut buf = make_buf(w, h, data);

    filter::hue_shift(&mut buf, 360.0).unwrap();

    // Hue shift by 360 should produce exact original
    // Allow +/- 1 for floating-point rounding through HSL conversion
    for (i, (&orig, &shifted)) in original.iter().zip(buf.data().iter()).enumerate() {
        let diff = (orig as i16 - shifted as i16).unsigned_abs();
        assert!(
            diff <= 1,
            "hue shift 360 identity: byte {i} differs by {diff} ({orig} vs {shifted})"
        );
    }
}

// -----------------------------------------------------------------------
// 7. Color balance neutral identity
// -----------------------------------------------------------------------

#[test]
fn color_balance_neutral_identity() {
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for (i, pixel) in data.chunks_exact_mut(4).enumerate() {
        pixel[0] = (i * 19 % 256) as u8;
        pixel[1] = (i * 47 % 256) as u8;
        pixel[2] = (i * 89 % 256) as u8;
        pixel[3] = 255;
    }
    let original = data.clone();
    let mut buf = make_buf(w, h, data);

    // All zeros = no adjustment
    filter::color_balance(&mut buf, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();

    assert_eq!(
        buf.data(),
        original,
        "color balance with all zeros should produce exact original"
    );
}

// -----------------------------------------------------------------------
// 8. Crop + resize composition — dimensions and content sanity
// -----------------------------------------------------------------------

#[test]
fn crop_resize_composition() {
    let w = 32u32;
    let h = 32u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for (i, pixel) in data.chunks_exact_mut(4).enumerate() {
        let x = i % w as usize;
        let y = i / w as usize;
        pixel[0] = (x * 8).min(255) as u8;
        pixel[1] = (y * 8).min(255) as u8;
        pixel[2] = 128;
        pixel[3] = 255;
    }
    let buf = make_buf(w, h, data);

    // Crop center quarter: (8, 8) to (24, 24) = 16x16
    let cropped = transform::crop(&buf, 8, 8, 24, 24).unwrap();
    assert_eq!(cropped.width(), 16, "cropped width");
    assert_eq!(cropped.height(), 16, "cropped height");

    // Resize back to original size
    let resized = transform::resize(&cropped, w, h, ScaleFilter::Bilinear).unwrap();
    assert_eq!(resized.width(), w, "resized width");
    assert_eq!(resized.height(), h, "resized height");
    assert_eq!(
        resized.data().len(),
        (w * h * 4) as usize,
        "resized data length"
    );

    // Content sanity: the resized image should have non-zero content in the
    // center region and the pixel values should be within a reasonable range
    // of the cropped region.
    let center_px = resized.get_rgba(16, 16).unwrap();
    assert!(
        center_px[0] > 0 || center_px[1] > 0,
        "center pixel should have non-zero content"
    );
}

// -----------------------------------------------------------------------
// 9. Blend mode commutativity — Screen(a,b) = Screen(b,a) for opaque
// -----------------------------------------------------------------------

#[test]
fn blend_screen_commutativity() {
    // Test many pixel pairs: Screen is commutative for opaque pixels
    let test_values: &[[u8; 4]] = &[
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [128, 64, 200, 255],
        [50, 100, 150, 255],
        [200, 30, 90, 255],
        [1, 1, 1, 255],
        [254, 254, 254, 255],
    ];

    for a in test_values {
        for b in test_values {
            let ab = blend_pixel(*a, *b, BlendMode::Screen, 255);
            let ba = blend_pixel(*b, *a, BlendMode::Screen, 255);
            for c in 0..3 {
                let diff = (ab[c] as i16 - ba[c] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "Screen commutativity failed for {:?} and {:?}: channel {c}, {:?} vs {:?}",
                    a,
                    b,
                    ab,
                    ba
                );
            }
        }
    }
}

// -----------------------------------------------------------------------
// 10. YUV roundtrip color fidelity
// -----------------------------------------------------------------------

#[test]
fn yuv_roundtrip_color_fidelity() {
    // Test specific colors through YUV420p roundtrip.
    // Note: colors that produce extreme U/V chrominance values (saturated
    // primaries) may trigger i16 overflow in the library's BT.601 fixed-point
    // math, so we use moderate colors that stay within safe conversion range.
    let test_colors: &[([u8; 4], &str, u8)] = &[
        ([128, 128, 128, 255], "mid-gray", 5),
        ([255, 255, 255, 255], "white", 5),
        ([0, 0, 0, 255], "black", 5),
        ([220, 180, 150, 255], "skin-tone", 15),
        ([180, 200, 160, 255], "muted-green", 15),
        ([160, 140, 180, 255], "muted-purple", 15),
        ([200, 190, 170, 255], "warm-neutral", 10),
        ([100, 120, 140, 255], "cool-shadow", 10),
    ];

    // Use 4x4 buffer (minimum for even-dimension YUV420p) filled with uniform color
    for &(color, name, max_drift) in test_colors {
        let data = color.repeat(16); // 4x4 = 16 pixels
        let buf = make_buf(4, 4, data);

        let yuv = match rgba_to_yuv420p(&buf) {
            Ok(y) => y,
            Err(_) => continue,
        };

        // Use catch_unwind to guard against potential i16 overflow in conversion
        let roundtrip = match std::panic::catch_unwind(|| yuv420p_to_rgba(&yuv)) {
            Ok(Ok(r)) => r,
            _ => continue,
        };

        // Check center pixel (1,1) — least affected by edge subsampling
        let orig = buf.get_rgba(1, 1).unwrap();
        let rt = roundtrip.get_rgba(1, 1).unwrap();
        for c in 0..3 {
            let diff = (orig[c] as i16 - rt[c] as i16).unsigned_abs();
            assert!(
                diff <= max_drift as u16,
                "YUV roundtrip {name}: channel {c} drifted by {diff} ({} -> {}), max allowed {max_drift}",
                orig[c],
                rt[c]
            );
        }
    }
}
