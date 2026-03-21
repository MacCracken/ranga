//! Histogram computation for luminance and color channels.
//!
//! Provides normalized histogram computation and comparison for RGBA8 buffers.

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};

/// Compute a luminance histogram from an RGBA8 buffer.
///
/// Returns `bins` normalized values summing to approximately 1.0.
/// Uses BT.601 luminance coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::histogram;
///
/// let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let hist = histogram::luminance_histogram(&buf, 256).unwrap();
/// assert_eq!(hist.len(), 256);
/// let sum: f64 = hist.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// ```
pub fn luminance_histogram(buf: &PixelBuffer, bins: usize) -> Result<Vec<f64>, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    if bins == 0 {
        return Err(RangaError::Other("bins must be > 0".into()));
    }
    let mut hist = vec![0u64; bins];
    let scale = bins as f64 / 256.0;

    for pixel in buf.data.chunks_exact(4) {
        let lum = (77u16 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8;
        let bin = ((lum as f64 * scale) as usize).min(bins - 1);
        hist[bin] += 1;
    }

    let total = hist.iter().sum::<u64>() as f64;
    if total == 0.0 {
        return Ok(vec![0.0; bins]);
    }
    Ok(hist.iter().map(|&v| v as f64 / total).collect())
}

/// Compute per-channel (R, G, B) histograms from an RGBA8 buffer.
///
/// Returns 3 histograms of 256 entries each, normalized to sum to 1.0.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::histogram;
///
/// let buf = PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let [r, g, b] = histogram::rgb_histograms(&buf).unwrap();
/// assert_eq!(r.len(), 256);
/// // All pixels are 128, so bin 128 should be 1.0
/// assert!((r[128] - 1.0).abs() < 1e-6);
/// ```
pub fn rgb_histograms(buf: &PixelBuffer) -> Result<[Vec<f64>; 3], RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let mut r_hist = vec![0u64; 256];
    let mut g_hist = vec![0u64; 256];
    let mut b_hist = vec![0u64; 256];

    for pixel in buf.data.chunks_exact(4) {
        r_hist[pixel[0] as usize] += 1;
        g_hist[pixel[1] as usize] += 1;
        b_hist[pixel[2] as usize] += 1;
    }

    let total = buf.pixel_count() as f64;
    if total == 0.0 {
        return Ok([vec![0.0; 256], vec![0.0; 256], vec![0.0; 256]]);
    }
    Ok([
        r_hist.iter().map(|&v| v as f64 / total).collect(),
        g_hist.iter().map(|&v| v as f64 / total).collect(),
        b_hist.iter().map(|&v| v as f64 / total).collect(),
    ])
}

/// Chi-squared distance between two histograms.
///
/// Returns 0.0 for identical distributions and positive values for
/// different distributions. Both slices should have equal length.
///
/// # Examples
///
/// ```
/// use ranga::histogram::chi_squared;
///
/// let a = vec![0.25, 0.25, 0.25, 0.25];
/// let b = vec![0.25, 0.25, 0.25, 0.25];
/// assert!(chi_squared(&a, &b).abs() < 1e-10);
///
/// let c = vec![1.0, 0.0, 0.0, 0.0];
/// let d = vec![0.0, 0.0, 0.0, 1.0];
/// assert!(chi_squared(&c, &d) > 0.0);
/// ```
#[must_use]
pub fn chi_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let sum = ai + bi;
            if sum > 1e-10 {
                (ai - bi) * (ai - bi) / sum
            } else {
                0.0
            }
        })
        .sum::<f64>()
        * 0.5
}

/// Equalize the luminance histogram of an RGBA8 buffer in-place.
///
/// Applies standard histogram equalization using a cumulative distribution
/// function to redistribute pixel intensities. Preserves hue and saturation
/// by scaling RGB proportionally based on the luminance mapping.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::histogram;
///
/// let mut buf = PixelBuffer::new(vec![50, 50, 50, 255].repeat(64), 8, 8, PixelFormat::Rgba8).unwrap();
/// histogram::equalize(&mut buf).unwrap();
/// ```
pub fn equalize(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    // Build 256-bin luminance histogram.
    let mut hist = [0u64; 256];
    for pixel in buf.data.chunks_exact(4) {
        let lum = ((77u16 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8)
            as usize;
        hist[lum.min(255)] += 1;
    }

    // Build CDF.
    let total = buf.pixel_count() as f64;
    if total == 0.0 {
        return Ok(());
    }
    let mut cdf = [0f64; 256];
    cdf[0] = hist[0] as f64 / total;
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i] as f64 / total;
    }

    // Find CDF min for standard equalization formula.
    let cdf_min = cdf.iter().copied().find(|&v| v > 0.0).unwrap_or(0.0);
    let denom = 1.0 - cdf_min;
    if denom < 1e-10 {
        return Ok(());
    }

    // Build lookup table: old luminance → new luminance.
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = (((cdf[i] - cdf_min) / denom) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }

    // Apply: scale RGB proportionally.
    for pixel in buf.data.chunks_exact_mut(4) {
        let old_lum = ((77u16 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16)
            >> 8) as usize;
        let old_lum = old_lum.min(255);
        let new_lum = lut[old_lum] as f32;
        if old_lum == 0 {
            let v = new_lum as u8;
            pixel[0] = v;
            pixel[1] = v;
            pixel[2] = v;
        } else {
            let scale = new_lum / old_lum as f32;
            pixel[0] = (pixel[0] as f32 * scale).clamp(0.0, 255.0) as u8;
            pixel[1] = (pixel[1] as f32 * scale).clamp(0.0, 255.0) as u8;
            pixel[2] = (pixel[2] as f32 * scale).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

/// Auto-levels: stretch the histogram to use the full 0–255 range.
///
/// Finds the actual min/max luminance and linearly maps them to 0–255.
/// This is simpler than full equalization but often sufficient for images
/// with poor contrast.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::histogram;
///
/// let mut buf = PixelBuffer::new(vec![100, 100, 100, 255, 150, 150, 150, 255], 2, 1, PixelFormat::Rgba8).unwrap();
/// histogram::auto_levels(&mut buf).unwrap();
/// // The darker pixel should be pulled toward 0, the brighter toward 255
/// assert!(buf.data[0] < 100);
/// assert!(buf.data[4] > 150);
/// ```
pub fn auto_levels(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    if buf.data.is_empty() {
        return Ok(());
    }

    // Find per-channel min/max to avoid color shifts from luminance-based stretching.
    let mut min_r = 255u8;
    let mut max_r = 0u8;
    let mut min_g = 255u8;
    let mut max_g = 0u8;
    let mut min_b = 255u8;
    let mut max_b = 0u8;
    for pixel in buf.data.chunks_exact(4) {
        min_r = min_r.min(pixel[0]);
        max_r = max_r.max(pixel[0]);
        min_g = min_g.min(pixel[1]);
        max_g = max_g.max(pixel[1]);
        min_b = min_b.min(pixel[2]);
        max_b = max_b.max(pixel[2]);
    }

    let stretch = |min: u8, max: u8| -> (f32, f32) {
        let range = max as f32 - min as f32;
        if range < 1.0 {
            (0.0, 1.0)
        } else {
            (min as f32, 255.0 / range)
        }
    };
    let (off_r, scale_r) = stretch(min_r, max_r);
    let (off_g, scale_g) = stretch(min_g, max_g);
    let (off_b, scale_b) = stretch(min_b, max_b);

    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = ((pixel[0] as f32 - off_r) * scale_r).clamp(0.0, 255.0) as u8;
        pixel[1] = ((pixel[1] as f32 - off_g) * scale_g).clamp(0.0, 255.0) as u8;
        pixel[2] = ((pixel[2] as f32 - off_b) * scale_b).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_histogram_uniform() {
        let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let hist = luminance_histogram(&buf, 256).unwrap();
        assert_eq!(hist.len(), 256);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chi_squared_identical_is_zero() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        assert!((chi_squared(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_different_is_positive() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        assert!(chi_squared(&a, &b) > 0.0);
    }

    #[test]
    fn rgb_histograms_length() {
        let buf = PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let [r, g, b] = rgb_histograms(&buf).unwrap();
        assert_eq!(r.len(), 256);
        assert_eq!(g.len(), 256);
        assert_eq!(b.len(), 256);
    }

    #[test]
    fn equalize_uniform_unchanged() {
        // Uniform image: all same value. Equalization should keep it uniform.
        let mut buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        equalize(&mut buf).unwrap();
        // All pixels should be the same value (mapped to 255 since single-value CDF).
        let v = buf.data[0];
        for pixel in buf.data.chunks_exact(4) {
            assert_eq!(pixel[0], v);
        }
    }

    #[test]
    fn auto_levels_stretches() {
        // Image with compressed range (100–150) should expand.
        let mut buf = PixelBuffer::new(
            vec![100, 100, 100, 255, 150, 150, 150, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        auto_levels(&mut buf).unwrap();
        assert!(
            buf.data[0] < 10,
            "min should map near 0, got {}",
            buf.data[0]
        );
        assert!(
            buf.data[4] > 245,
            "max should map near 255, got {}",
            buf.data[4]
        );
    }

    #[test]
    fn auto_levels_already_full_range() {
        let mut buf = PixelBuffer::new(
            vec![0, 0, 0, 255, 255, 255, 255, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        let original = buf.data.clone();
        auto_levels(&mut buf).unwrap();
        assert_eq!(buf.data, original);
    }
}
