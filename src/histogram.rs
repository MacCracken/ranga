//! Histogram computation for luminance and color channels.

use crate::pixel::{PixelBuffer, PixelFormat};
use crate::RangaError;

/// Compute a luminance histogram from an RGBA8 buffer.
/// Returns `bins` normalized values summing to approximately 1.0.
pub fn luminance_histogram(buf: &PixelBuffer, bins: usize) -> Result<Vec<f64>, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
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
/// Returns 3 histograms of 256 entries each, normalized.
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
}
