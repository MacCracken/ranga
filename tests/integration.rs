use ranga::RangaError;
use ranga::blend::{BlendMode, blend_pixel, blend_row_normal};
use ranga::convert::{argb_to_nv12, rgba_to_yuv420p, yuv420p_to_rgba};
use ranga::filter;
use ranga::histogram::{chi_squared, luminance_histogram, rgb_histograms};
use ranga::pixel::{PixelBuffer, PixelFormat};

// ---------------------------------------------------------------------------
// Pixel buffer roundtrips
// ---------------------------------------------------------------------------

#[test]
fn pixel_buffer_new_and_access() {
    let data = vec![10, 20, 30, 255, 40, 50, 60, 128];
    let buf = PixelBuffer::new(data.clone(), 2, 1, PixelFormat::Rgba8).unwrap();
    assert_eq!(buf.width, 2);
    assert_eq!(buf.height, 1);
    assert_eq!(buf.format, PixelFormat::Rgba8);
    assert_eq!(buf.data, data);
    assert_eq!(buf.pixel_count(), 2);
}

#[test]
fn pixel_buffer_zeroed_roundtrip() {
    let buf = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);
    assert_eq!(buf.data.len(), 8 * 8 * 4);
    assert!(buf.data.iter().all(|&b| b == 0));

    // Re-wrap via new to validate size consistency
    let buf2 = PixelBuffer::new(buf.data.clone(), 8, 8, PixelFormat::Rgba8).unwrap();
    assert_eq!(buf2.data, buf.data);
}

// ---------------------------------------------------------------------------
// Blend correctness
// ---------------------------------------------------------------------------

#[test]
fn blend_red_over_blue_full_opacity() {
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    let result = blend_pixel(red, blue, BlendMode::Normal, 255);
    // At full opacity, result should be mostly red
    assert!(
        result[0] > 200,
        "red channel should dominate: got {}",
        result[0]
    );
    assert!(
        result[2] < 60,
        "blue channel should be low: got {}",
        result[2]
    );
}

#[test]
fn blend_red_over_blue_half_opacity() {
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    let result = blend_pixel(red, blue, BlendMode::Normal, 128);
    // At ~50% opacity, both channels should be present
    assert!(
        result[0] > 50 && result[0] < 220,
        "red should be mid-range: {}",
        result[0]
    );
    assert!(
        result[2] > 50 && result[2] < 220,
        "blue should be mid-range: {}",
        result[2]
    );
}

#[test]
fn blend_red_over_blue_zero_opacity() {
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    let result = blend_pixel(red, blue, BlendMode::Normal, 0);
    // Zero opacity: destination unchanged
    assert_eq!(result, blue);
}

#[test]
fn blend_row_normal_produces_correct_length() {
    let src = vec![[255u8, 0, 0, 255]; 4]
        .into_iter()
        .flatten()
        .collect::<Vec<u8>>();
    let mut dst = vec![[0u8, 0, 255, 255]; 4]
        .into_iter()
        .flatten()
        .collect::<Vec<u8>>();
    let original_len = dst.len();
    blend_row_normal(&src, &mut dst, 200);
    assert_eq!(dst.len(), original_len);
    // Every pixel should have been modified (red blended over blue)
    for pixel in dst.chunks_exact(4) {
        assert!(pixel[0] > 100, "expected red influence");
    }
}

// ---------------------------------------------------------------------------
// Filter pipelines
// ---------------------------------------------------------------------------

#[test]
fn filter_pipeline_brightness_contrast_grayscale() {
    // Create a buffer with varied pixel values
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for i in 0..16u8 {
        let v = i * 16;
        data.extend_from_slice(&[v, v.wrapping_add(30), v.wrapping_add(60), 255]);
    }
    let mut buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();

    filter::brightness(&mut buf, 0.1).unwrap();
    filter::contrast(&mut buf, 1.2).unwrap();
    filter::grayscale(&mut buf).unwrap();

    // After grayscale, R == G == B for every pixel
    for pixel in buf.data.chunks_exact(4) {
        assert_eq!(pixel[0], pixel[1], "R != G after grayscale");
        assert_eq!(pixel[1], pixel[2], "G != B after grayscale");
        assert_eq!(pixel[3], 255, "alpha should be preserved");
    }
}

#[test]
fn filter_invert_double_is_identity() {
    let original_data = vec![100, 150, 200, 255, 10, 20, 30, 128];
    let mut buf = PixelBuffer::new(original_data.clone(), 2, 1, PixelFormat::Rgba8).unwrap();
    filter::invert(&mut buf).unwrap();
    filter::invert(&mut buf).unwrap();
    assert_eq!(buf.data, original_data);
}

// ---------------------------------------------------------------------------
// Color conversion roundtrip
// ---------------------------------------------------------------------------

#[test]
fn rgba_yuv420p_roundtrip_within_tolerance() {
    // Use even dimensions for chroma subsampling.
    // Use a uniform-color buffer so chroma subsampling doesn't introduce
    // large errors at block boundaries.
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        data[i * 4] = 128;
        data[i * 4 + 1] = 100;
        data[i * 4 + 2] = 80;
        data[i * 4 + 3] = 255;
    }

    let original = data.clone();
    let buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();
    let yuv = rgba_to_yuv420p(&buf).unwrap();
    assert_eq!(yuv.format, PixelFormat::Yuv420p);

    let back = yuv420p_to_rgba(&yuv).unwrap();
    assert_eq!(back.format, PixelFormat::Rgba8);
    assert_eq!(back.data.len(), original.len());

    // YUV conversion is lossy; check per-channel tolerance
    let max_diff = original
        .chunks_exact(4)
        .zip(back.data.chunks_exact(4))
        .flat_map(|(o, r)| (0..3).map(move |c| (o[c] as i16 - r[c] as i16).unsigned_abs()))
        .max()
        .unwrap_or(0);

    assert!(
        max_diff < 20,
        "max per-channel diff was {max_diff}, expected < 20"
    );
}

// ---------------------------------------------------------------------------
// Histogram on known data
// ---------------------------------------------------------------------------

#[test]
fn histogram_uniform_buffer_peaks_at_one_bin() {
    // All pixels identical (128,128,128,255) — luminance should land in one bin
    let buf = PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
    let hist = luminance_histogram(&buf, 256).unwrap();
    assert_eq!(hist.len(), 256);

    // Exactly one bin should hold all the weight
    let nonzero_bins: Vec<_> = hist.iter().filter(|&&v| v > 0.0).collect();
    assert_eq!(nonzero_bins.len(), 1, "expected exactly one nonzero bin");
    assert!((nonzero_bins[0] - 1.0).abs() < 1e-6);
}

#[test]
fn rgb_histograms_pure_red_buffer() {
    let data: Vec<u8> = (0..16).flat_map(|_| [255, 0, 0, 255]).collect();
    let buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();
    let [r, g, b] = rgb_histograms(&buf).unwrap();

    // Red channel: all weight at bin 255
    assert!((r[255] - 1.0).abs() < 1e-6);
    // Green and blue: all weight at bin 0
    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((b[0] - 1.0).abs() < 1e-6);
}

#[test]
fn chi_squared_identical_histograms() {
    let a = vec![0.25, 0.25, 0.25, 0.25];
    let b = vec![0.25, 0.25, 0.25, 0.25];
    let d = chi_squared(&a, &b);
    assert!(
        d.abs() < 1e-10,
        "identical histograms should have chi^2 ~ 0"
    );
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn pixel_buffer_dimension_mismatch() {
    // 10 bytes is wrong for 2x2 RGBA8 (should be 16)
    let result = PixelBuffer::new(vec![0; 10], 2, 2, PixelFormat::Rgba8);
    assert!(result.is_err());
    match result.unwrap_err() {
        RangaError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 16);
            assert_eq!(actual, 10);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn filter_rejects_wrong_format() {
    let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgb8);
    assert!(filter::brightness(&mut buf, 0.1).is_err());
    assert!(filter::contrast(&mut buf, 1.0).is_err());
    assert!(filter::grayscale(&mut buf).is_err());
    assert!(filter::invert(&mut buf).is_err());
}

#[test]
fn convert_rejects_wrong_format() {
    let rgba_buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
    // yuv420p_to_rgba expects Yuv420p, not Rgba8
    assert!(yuv420p_to_rgba(&rgba_buf).is_err());

    let rgb_buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgb8);
    // rgba_to_yuv420p expects Rgba8, not Rgb8
    assert!(rgba_to_yuv420p(&rgb_buf).is_err());

    // argb_to_nv12 expects Argb8
    assert!(argb_to_nv12(&rgba_buf).is_err());
}

// ---------------------------------------------------------------------------
// Transform integration tests
// ---------------------------------------------------------------------------

#[test]
fn crop_and_resize_pipeline() {
    use ranga::transform::{self, ScaleFilter};

    let buf = PixelBuffer::zeroed(100, 100, PixelFormat::Rgba8);
    let cropped = transform::crop(&buf, 10, 10, 60, 60).unwrap();
    assert_eq!(cropped.width, 50);
    assert_eq!(cropped.height, 50);
    let resized = transform::resize(&cropped, 200, 200, ScaleFilter::Bilinear).unwrap();
    assert_eq!(resized.width, 200);
    assert_eq!(resized.height, 200);
}

#[test]
fn affine_rotate_and_back() {
    use ranga::transform::{self, Affine, ScaleFilter};

    let mut buf = PixelBuffer::zeroed(32, 32, PixelFormat::Rgba8);
    buf.set_rgba(16, 16, [255, 0, 0, 255]);

    let rotated =
        transform::affine_transform(&buf, &Affine::rotate(0.1), 32, 32, ScaleFilter::Bilinear)
            .unwrap();
    assert_eq!(rotated.width, 32);
    // The red pixel should have moved slightly
}

#[test]
fn flip_preserves_pixel_count() {
    let data: Vec<u8> = (0..64u8).flat_map(|i| [i, i, i, 255]).collect();
    let buf = PixelBuffer::new(data, 8, 8, PixelFormat::Rgba8).unwrap();
    let fh = ranga::transform::flip_horizontal(&buf).unwrap();
    let fv = ranga::transform::flip_vertical(&buf).unwrap();
    assert_eq!(fh.data.len(), buf.data.len());
    assert_eq!(fv.data.len(), buf.data.len());
}

// ---------------------------------------------------------------------------
// Composite integration tests
// ---------------------------------------------------------------------------

#[test]
fn compositor_pipeline() {
    use ranga::composite;

    // Simulate aethersafta workflow: checkerboard bg → composite layer → fade
    let mut bg = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    composite::fill_checkerboard(&mut bg, 8, [200, 200, 200, 255], [255, 255, 255, 255]).unwrap();

    let overlay =
        PixelBuffer::new([255, 0, 0, 128].repeat(16 * 16), 16, 16, PixelFormat::Rgba8).unwrap();
    composite::composite_at(&overlay, &mut bg, 24, 24, 0.8).unwrap();

    // Pixel at (24,24) should have red content blended over checkerboard
    let px = bg.get_rgba(24, 24).unwrap();
    assert!(px[0] > 100, "red channel should be present");
}

#[test]
fn dissolve_then_fade_pipeline() {
    use ranga::composite;

    let a = PixelBuffer::new(vec![200; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
    let b = PixelBuffer::new(vec![50; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();

    let mut mid = composite::dissolve(&a, &b, 0.5).unwrap();
    assert!(mid.data[0] > 100 && mid.data[0] < 150);

    composite::fade(&mut mid, 0.5).unwrap();
    assert!(mid.data[0] < 80); // faded to half
}

#[test]
fn premultiply_composite_unpremultiply() {
    use ranga::composite;

    let mut layer = PixelBuffer::new(vec![255, 0, 0, 128], 1, 1, PixelFormat::Rgba8).unwrap();
    composite::premultiply_alpha(&mut layer).unwrap();
    assert_eq!(layer.data[0], 128); // 255 * 128 / 255

    composite::unpremultiply_alpha(&mut layer).unwrap();
    assert!(layer.data[0] > 250); // back to ~255
}

// ---------------------------------------------------------------------------
// New filter integration tests
// ---------------------------------------------------------------------------

#[test]
fn median_removes_salt_pepper() {
    let mut buf = PixelBuffer::new(vec![128; 16 * 16 * 4], 16, 16, PixelFormat::Rgba8).unwrap();
    filter::noise_salt_pepper(&mut buf, 0.3, 42).unwrap();

    let noisy_extremes = buf
        .data
        .chunks_exact(4)
        .filter(|p| p[0] == 0 || p[0] == 255)
        .count();
    assert!(noisy_extremes > 0, "should have noise");

    let filtered = filter::median(&buf, 1).unwrap();
    let clean_extremes = filtered
        .data
        .chunks_exact(4)
        .filter(|p| p[0] == 0 || p[0] == 255)
        .count();
    assert!(
        clean_extremes < noisy_extremes,
        "median should reduce extremes"
    );
}

#[test]
fn flood_fill_then_threshold() {
    let mut buf = PixelBuffer::new(vec![100; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
    filter::flood_fill(&mut buf, 0, 0, [200, 200, 200, 255], 10).unwrap();
    assert_eq!(buf.data[0], 200);

    filter::threshold(&mut buf, 150).unwrap();
    assert_eq!(buf.data[0], 255); // 200 > 150 → white
}
