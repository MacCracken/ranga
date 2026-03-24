//! Edge-case tests for ranga — coverage gap fillers.

use ranga::RangaError;
use ranga::blend::{BlendMode, blend_pixel, blend_pixel_argb, blend_row_normal_argb};
use ranga::composite;
use ranga::convert;
use ranga::filter;
use ranga::histogram;
use ranga::pixel::{BufferPool, PixelBuffer, PixelFormat, PixelView};
use ranga::transform::{self, Perspective, ScaleFilter};

// =========================================================================
// 1. Error variant formatting
// =========================================================================

#[test]
fn error_invalid_format_display() {
    let err = RangaError::InvalidFormat("BGRA".into());
    let msg = err.to_string();
    assert!(
        msg.contains("invalid pixel format") && msg.contains("BGRA"),
        "unexpected message: {msg}"
    );
}

#[test]
fn error_dimension_mismatch_display() {
    let err = RangaError::DimensionMismatch {
        expected: 1024,
        actual: 512,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("dimension mismatch") && msg.contains("1024") && msg.contains("512"),
        "unexpected message: {msg}"
    );
}

#[test]
fn error_buffer_too_small_display() {
    let err = RangaError::BufferTooSmall {
        need: 2048,
        have: 1024,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("buffer too small") && msg.contains("2048") && msg.contains("1024"),
        "unexpected message: {msg}"
    );
}

#[test]
fn error_unsupported_conversion_display() {
    let err = RangaError::UnsupportedConversion {
        from: "NV12".into(),
        to: "RgbaF32".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported conversion") && msg.contains("NV12") && msg.contains("RgbaF32"),
        "unexpected message: {msg}"
    );
}

#[test]
fn error_other_display() {
    let err = RangaError::Other("something went wrong".into());
    assert_eq!(err.to_string(), "something went wrong");
}

// =========================================================================
// 2. ARGB blend tests
// =========================================================================

#[test]
fn blend_pixel_argb_opaque_matches_rgba() {
    // Opaque red over opaque blue — ARGB layout [A, R, G, B]
    let src_argb = [255, 255, 0, 0]; // opaque red
    let dst_argb = [255, 0, 0, 255]; // opaque blue

    let result_argb = blend_pixel_argb(src_argb, dst_argb, BlendMode::Normal, 255);

    // Equivalent RGBA blend
    let src_rgba = [255, 0, 0, 255];
    let dst_rgba = [0, 0, 255, 255];
    let result_rgba = blend_pixel(src_rgba, dst_rgba, BlendMode::Normal, 255);

    // ARGB result [A, R, G, B] should match RGBA result [R, G, B, A]
    assert_eq!(result_argb[0], result_rgba[3], "alpha mismatch");
    assert_eq!(result_argb[1], result_rgba[0], "red mismatch");
    assert_eq!(result_argb[2], result_rgba[1], "green mismatch");
    assert_eq!(result_argb[3], result_rgba[2], "blue mismatch");
}

#[test]
fn blend_row_normal_argb_preserves_length() {
    let src = vec![200, 128, 64, 32, 200, 64, 128, 96]; // 2 ARGB pixels
    let mut dst = vec![255, 0, 0, 255, 255, 0, 0, 255];
    blend_row_normal_argb(&src, &mut dst, 255);
    assert_eq!(dst.len(), 8);
}

#[test]
fn blend_row_normal_argb_zero_opacity_noop() {
    let src = vec![255, 255, 0, 0, 255, 0, 255, 0]; // 2 opaque ARGB pixels
    let mut dst = vec![255, 0, 0, 255, 255, 100, 100, 100];
    let original = dst.clone();
    blend_row_normal_argb(&src, &mut dst, 0);
    assert_eq!(dst, original, "zero opacity should leave dst unchanged");
}

#[test]
fn blend_pixel_argb_transparent_source_noop() {
    let src = [0, 255, 0, 0]; // transparent source (A=0)
    let dst = [255, 0, 0, 255]; // opaque blue
    let result = blend_pixel_argb(src, dst, BlendMode::Normal, 255);
    assert_eq!(result, dst, "transparent source should leave dst unchanged");
}

// =========================================================================
// 3. composite_at_argb tests
// =========================================================================

#[test]
fn composite_at_argb_basic() -> Result<(), RangaError> {
    let mut dst = PixelBuffer::zeroed(10, 10, PixelFormat::Argb8);
    // 2x2 red pixels: [A=255, R=255, G=0, B=0]
    let src = PixelBuffer::new([255, 255, 0, 0].repeat(4), 2, 2, PixelFormat::Argb8)?;
    composite::composite_at_argb(&src, &mut dst, 3, 3, 1.0)?;

    // Check pixel at (3,3) — should be opaque red
    let idx = (3 * 10 + 3) * 4;
    assert_eq!(dst.data[idx], 255, "alpha should be 255");
    assert_eq!(dst.data[idx + 1], 255, "red should be 255");
    assert_eq!(dst.data[idx + 2], 0, "green should be 0");
    assert_eq!(dst.data[idx + 3], 0, "blue should be 0");
    Ok(())
}

#[test]
fn composite_at_argb_clipping() -> Result<(), RangaError> {
    let mut dst = PixelBuffer::zeroed(4, 4, PixelFormat::Argb8);
    let src = PixelBuffer::new([255, 128, 64, 32].repeat(9), 3, 3, PixelFormat::Argb8)?;
    // Place at (2,2) so only a 2x2 area overlaps
    composite::composite_at_argb(&src, &mut dst, 2, 2, 1.0)?;

    // Pixel at (2,2) should be composited
    let idx = (2 * 4 + 2) * 4;
    assert!(
        dst.data[idx] > 0,
        "overlapping pixel should be non-zero alpha"
    );

    // Pixel at (0,0) should still be zero
    assert_eq!(dst.data[0], 0, "non-overlapping pixel should remain zero");
    Ok(())
}

#[test]
fn composite_at_argb_wrong_format_rejected() {
    let mut dst = PixelBuffer::zeroed(4, 4, PixelFormat::Argb8);
    let src = PixelBuffer::zeroed(2, 2, PixelFormat::Rgba8); // wrong format
    let result = composite::composite_at_argb(&src, &mut dst, 0, 0, 1.0);
    assert!(
        result.is_err(),
        "RGBA8 source should be rejected for ARGB composite"
    );
}

// =========================================================================
// 4. Histogram edge cases
// =========================================================================

#[test]
fn equalize_gradient_expands_range() -> Result<(), RangaError> {
    // Create a gradient from 50 to 200
    let mut data = Vec::with_capacity(256 * 4);
    for i in 0..256u16 {
        let v = (50.0 + (i as f32 / 255.0) * 150.0) as u8;
        data.extend_from_slice(&[v, v, v, 255]);
    }
    let mut buf = PixelBuffer::new(data, 256, 1, PixelFormat::Rgba8)?;

    let min_before = buf
        .data
        .chunks_exact(4)
        .map(|p| p[0])
        .min()
        .expect("non-empty");
    let max_before = buf
        .data
        .chunks_exact(4)
        .map(|p| p[0])
        .max()
        .expect("non-empty");

    histogram::equalize(&mut buf)?;

    let min_after = buf
        .data
        .chunks_exact(4)
        .map(|p| p[0])
        .min()
        .expect("non-empty");
    let max_after = buf
        .data
        .chunks_exact(4)
        .map(|p| p[0])
        .max()
        .expect("non-empty");

    let range_before = max_before - min_before;
    let range_after = max_after - min_after;
    assert!(
        range_after >= range_before,
        "equalize should expand range: before={range_before}, after={range_after}"
    );
    Ok(())
}

#[test]
fn auto_levels_single_value_unchanged() -> Result<(), RangaError> {
    let mut buf = PixelBuffer::new([100, 100, 100, 255].repeat(16), 4, 4, PixelFormat::Rgba8)?;
    let original = buf.data.clone();
    histogram::auto_levels(&mut buf)?;
    // Single value image: range is 0, so stretch does (0.0, 1.0) — identity
    assert_eq!(
        buf.data, original,
        "single-value image should be unchanged by auto_levels"
    );
    Ok(())
}

#[test]
fn luminance_histogram_various_bin_counts() -> Result<(), RangaError> {
    let buf = PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8)?;

    let h1 = histogram::luminance_histogram(&buf, 1)?;
    assert_eq!(h1.len(), 1);
    assert!((h1[0] - 1.0).abs() < 1e-6, "single bin should be 1.0");

    let h10 = histogram::luminance_histogram(&buf, 10)?;
    assert_eq!(h10.len(), 10);
    let sum10: f64 = h10.iter().sum();
    assert!((sum10 - 1.0).abs() < 1e-6);

    let h256 = histogram::luminance_histogram(&buf, 256)?;
    assert_eq!(h256.len(), 256);
    let sum256: f64 = h256.iter().sum();
    assert!((sum256 - 1.0).abs() < 1e-6);

    Ok(())
}

// =========================================================================
// 5. Filter edge cases
// =========================================================================

#[test]
fn gaussian_blur_large_radius_small_image_no_panic() -> Result<(), RangaError> {
    let buf = PixelBuffer::new(vec![128; 2 * 2 * 4], 2, 2, PixelFormat::Rgba8)?;
    let result = filter::gaussian_blur(&buf, 50)?;
    assert_eq!(result.width, 2);
    assert_eq!(result.height, 2);
    Ok(())
}

#[test]
fn box_blur_radius_zero_identity() -> Result<(), RangaError> {
    let buf = PixelBuffer::new([100, 150, 200, 255].repeat(16), 4, 4, PixelFormat::Rgba8)?;
    let result = filter::box_blur(&buf, 0)?;
    assert_eq!(result.data, buf.data, "radius 0 should be identity");
    Ok(())
}

#[test]
fn auto_white_balance_balanced_is_near_identity() -> Result<(), RangaError> {
    // Already balanced: all channels have same average (128)
    let mut buf = PixelBuffer::new([128, 128, 128, 255].repeat(64), 8, 8, PixelFormat::Rgba8)?;
    let original = buf.data.clone();
    filter::auto_white_balance(&mut buf)?;
    // Each channel should remain very close to the original
    for (i, (&orig, &new)) in original.iter().zip(buf.data.iter()).enumerate() {
        if i % 4 != 3 {
            // skip alpha
            assert!(
                (orig as i16 - new as i16).unsigned_abs() <= 1,
                "pixel byte {i}: expected ~{orig}, got {new}"
            );
        }
    }
    Ok(())
}

#[test]
fn bilateral_single_color_identity() -> Result<(), RangaError> {
    let buf = PixelBuffer::new([100, 100, 100, 255].repeat(16), 4, 4, PixelFormat::Rgba8)?;
    let result = filter::bilateral(&buf, 2, 10.0, 30.0)?;
    // All pixels are the same color, so bilateral should not change them
    for (i, (&orig, &filtered)) in buf.data.iter().zip(result.data.iter()).enumerate() {
        assert_eq!(orig, filtered, "byte {i} changed from {orig} to {filtered}");
    }
    Ok(())
}

#[test]
fn vibrance_negative_reduces_saturation() -> Result<(), RangaError> {
    // Use a partially saturated pixel so vibrance has room to act.
    // Vibrance scales by (1.0 - sat), so fully saturated pixels (sat=1.0) are unchanged.
    let mut buf = PixelBuffer::new(vec![200, 100, 150, 255], 1, 1, PixelFormat::Rgba8)?;
    let orig_r = buf.data[0] as i16;
    let orig_g = buf.data[1] as i16;
    let orig_b = buf.data[2] as i16;
    let orig_spread = (orig_r - orig_g).unsigned_abs()
        + (orig_r - orig_b).unsigned_abs()
        + (orig_g - orig_b).unsigned_abs();

    filter::vibrance(&mut buf, -0.8)?;

    let r = buf.data[0] as i16;
    let g = buf.data[1] as i16;
    let b = buf.data[2] as i16;
    let spread = (r - g).unsigned_abs() + (r - b).unsigned_abs() + (g - b).unsigned_abs();
    assert!(
        spread < orig_spread,
        "negative vibrance should reduce saturation: orig_spread={orig_spread}, spread={spread}"
    );
    Ok(())
}

// =========================================================================
// 6. Pixel edge cases
// =========================================================================

#[test]
fn get_set_rgba_roundtrip() {
    let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
    let pixel = [42, 137, 200, 180];
    assert!(buf.set_rgba(2, 3, pixel));
    assert_eq!(buf.get_rgba(2, 3), Some(pixel));
}

#[test]
fn pixel_view_too_small_slice_returns_error() {
    let data = vec![0u8; 10]; // too small for 4x4 RGBA8 (needs 64)
    let result = PixelView::new(&data, 4, 4, PixelFormat::Rgba8);
    assert!(result.is_err(), "too-small slice should return error");
}

#[test]
fn buffer_pool_clear_empties() {
    let mut pool = BufferPool::new(8);
    pool.release(vec![0u8; 1024]);
    pool.release(vec![0u8; 2048]);
    assert_eq!(pool.len(), 2);
    pool.clear();
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
}

#[test]
fn buffer_pool_is_empty_initially() {
    let pool = BufferPool::new(4);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
}

// =========================================================================
// 7. Transform edge cases
// =========================================================================

#[test]
fn crop_zero_size_region() -> Result<(), RangaError> {
    let buf = PixelBuffer::zeroed(10, 10, PixelFormat::Rgba8);
    // left == right => zero width
    let cropped = transform::crop(&buf, 5, 5, 5, 5)?;
    assert_eq!(cropped.width, 0);
    assert_eq!(cropped.height, 0);
    assert!(cropped.data.is_empty());
    Ok(())
}

#[test]
fn resize_to_1x1() -> Result<(), RangaError> {
    // Fill with a solid color
    let buf = PixelBuffer::new([100, 150, 200, 255].repeat(64), 8, 8, PixelFormat::Rgba8)?;
    let small = transform::resize(&buf, 1, 1, ScaleFilter::Bilinear)?;
    assert_eq!(small.width, 1);
    assert_eq!(small.height, 1);
    assert_eq!(small.data.len(), 4);
    // Should sample somewhere near the original color
    assert!(
        (small.data[0] as i16 - 100).unsigned_abs() < 20,
        "red channel unexpected: {}",
        small.data[0]
    );
    Ok(())
}

#[test]
fn perspective_identity_preserves_image() -> Result<(), RangaError> {
    // Use a solid-color image to avoid offset/interpolation artifacts
    let buf = PixelBuffer::new([100, 150, 200, 255].repeat(64), 8, 8, PixelFormat::Rgba8)?;
    let p = Perspective::identity();
    let result = transform::perspective_transform(&buf, &p, 8, 8)?;
    assert_eq!(result.width, 8);
    assert_eq!(result.height, 8);
    // Solid color should be preserved everywhere
    for (i, (&orig, &out)) in buf.data.iter().zip(result.data.iter()).enumerate() {
        assert!(
            (orig as i16 - out as i16).unsigned_abs() <= 1,
            "byte {i}: expected {orig}, got {out}"
        );
    }
    Ok(())
}

// =========================================================================
// 8. Convert edge cases
// =========================================================================

#[test]
fn rgba_to_yuv420p_bt2020_roundtrip() -> Result<(), RangaError> {
    let buf = PixelBuffer::new([128, 128, 128, 255].repeat(16), 4, 4, PixelFormat::Rgba8)?;
    let yuv = convert::rgba_to_yuv420p_bt2020(&buf)?;
    assert_eq!(yuv.format, PixelFormat::Yuv420p);
    let back = convert::yuv420p_to_rgba_bt2020(&yuv)?;
    assert_eq!(back.format, PixelFormat::Rgba8);
    // Roundtrip for gray should be close
    for (i, pixel) in back.data.chunks_exact(4).enumerate() {
        for (c, &val) in pixel.iter().enumerate().take(3) {
            assert!(
                (val as i16 - 128).unsigned_abs() < 10,
                "pixel {i} channel {c}: expected ~128, got {val}",
            );
        }
        assert_eq!(pixel[3], 255, "alpha should be 255");
    }
    Ok(())
}

#[test]
fn rgba8_to_argb8_roundtrip() -> Result<(), RangaError> {
    let original = PixelBuffer::new(
        vec![200, 100, 50, 220, 10, 20, 30, 128],
        2,
        1,
        PixelFormat::Rgba8,
    )?;
    let argb = convert::rgba8_to_argb8(&original)?;
    assert_eq!(argb.format, PixelFormat::Argb8);
    let back = convert::argb8_to_rgba8(&argb)?;
    assert_eq!(back.format, PixelFormat::Rgba8);
    assert_eq!(
        back.data, original.data,
        "RGBA -> ARGB -> RGBA should be identity"
    );
    Ok(())
}

#[test]
fn rgba8_to_rgb8_then_back_sets_alpha_255() -> Result<(), RangaError> {
    let original = PixelBuffer::new(
        vec![200, 100, 50, 128, 10, 20, 30, 64],
        2,
        1,
        PixelFormat::Rgba8,
    )?;
    let rgb = convert::rgba8_to_rgb8(&original)?;
    assert_eq!(rgb.format, PixelFormat::Rgb8);
    let back = convert::rgb8_to_rgba8(&rgb)?;
    assert_eq!(back.format, PixelFormat::Rgba8);
    // RGB channels should match, alpha should be 255
    for (i, (orig_px, back_px)) in original
        .data
        .chunks_exact(4)
        .zip(back.data.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(orig_px[0], back_px[0], "pixel {i} R mismatch");
        assert_eq!(orig_px[1], back_px[1], "pixel {i} G mismatch");
        assert_eq!(orig_px[2], back_px[2], "pixel {i} B mismatch");
        assert_eq!(
            back_px[3], 255,
            "pixel {i} alpha should be 255, got {}",
            back_px[3]
        );
    }
    Ok(())
}
