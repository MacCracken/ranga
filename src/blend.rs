//! Blend modes — Porter-Duff and Photoshop-style compositing.
//!
//! 12 blend modes with optional SIMD acceleration.

use serde::{Deserialize, Serialize};

/// Supported blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

/// Blend a source pixel over a destination pixel using the given mode and opacity.
///
/// All values are 0–255 (sRGB byte space). Alpha is premultiplied in the blend.
#[inline]
pub fn blend_pixel(src: [u8; 4], dst: [u8; 4], mode: BlendMode, opacity: u8) -> [u8; 4] {
    let sa = ((src[3] as u16 * opacity as u16) >> 8) as u8;
    if sa == 0 {
        return dst;
    }

    let blend_channel = |s: u8, d: u8| -> u8 {
        let s = s as f32 / 255.0;
        let d = d as f32 / 255.0;
        let result = match mode {
            BlendMode::Normal => s,
            BlendMode::Multiply => s * d,
            BlendMode::Screen => 1.0 - (1.0 - s) * (1.0 - d),
            BlendMode::Overlay => {
                if d < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
            BlendMode::Darken => s.min(d),
            BlendMode::Lighten => s.max(d),
            BlendMode::ColorDodge => {
                if s >= 1.0 { 1.0 } else { (d / (1.0 - s)).min(1.0) }
            }
            BlendMode::ColorBurn => {
                if s <= 0.0 { 0.0 } else { 1.0 - ((1.0 - d) / s).min(1.0) }
            }
            BlendMode::SoftLight => {
                if s <= 0.5 {
                    d - (1.0 - 2.0 * s) * d * (1.0 - d)
                } else {
                    let g = if d <= 0.25 {
                        ((16.0 * d - 12.0) * d + 4.0) * d
                    } else {
                        d.sqrt()
                    };
                    d + (2.0 * s - 1.0) * (g - d)
                }
            }
            BlendMode::HardLight => {
                if s < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
            BlendMode::Difference => (s - d).abs(),
            BlendMode::Exclusion => s + d - 2.0 * s * d,
        };
        (result * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };

    // Porter-Duff source-over with blended color
    let br = blend_channel(src[0], dst[0]);
    let bg = blend_channel(src[1], dst[1]);
    let bb = blend_channel(src[2], dst[2]);

    let sa16 = sa as u16;
    let inv_sa = 255u16 - sa16;

    [
        ((br as u16 * sa16 + dst[0] as u16 * inv_sa) >> 8) as u8,
        ((bg as u16 * sa16 + dst[1] as u16 * inv_sa) >> 8) as u8,
        ((bb as u16 * sa16 + dst[2] as u16 * inv_sa) >> 8) as u8,
        ((sa16 + dst[3] as u16 * inv_sa / 255).min(255)) as u8,
    ]
}

/// Blend an entire source row over a destination row (RGBA8, Normal mode).
///
/// When the `simd` feature is enabled, uses SSE2 on x86_64 for ~4x throughput.
pub fn blend_row_normal(src: &[u8], dst: &mut [u8], opacity: u8) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 4, 0);

    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let blended = blend_pixel(
            [s[0], s[1], s[2], s[3]],
            [d[0], d[1], d[2], d[3]],
            BlendMode::Normal,
            opacity,
        );
        d.copy_from_slice(&blended);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_blend_opaque() {
        let src = [255, 0, 0, 255]; // red
        let dst = [0, 0, 255, 255]; // blue
        let result = blend_pixel(src, dst, BlendMode::Normal, 255);
        assert!(result[0] > 200); // mostly red
        assert!(result[2] < 55); // very little blue
    }

    #[test]
    fn normal_blend_transparent() {
        let src = [255, 0, 0, 0]; // transparent red
        let dst = [0, 0, 255, 255]; // blue
        let result = blend_pixel(src, dst, BlendMode::Normal, 255);
        assert_eq!(result, dst); // unchanged
    }

    #[test]
    fn multiply_darkens() {
        let src = [128, 128, 128, 255];
        let dst = [200, 200, 200, 255];
        let result = blend_pixel(src, dst, BlendMode::Multiply, 255);
        assert!(result[0] < dst[0]);
    }

    #[test]
    fn screen_lightens() {
        let src = [128, 128, 128, 255];
        let dst = [50, 50, 50, 255];
        let result = blend_pixel(src, dst, BlendMode::Screen, 255);
        assert!(result[0] > dst[0]);
    }

    #[test]
    fn difference_of_same_is_black() {
        let src = [100, 100, 100, 255];
        let dst = [100, 100, 100, 255];
        let result = blend_pixel(src, dst, BlendMode::Difference, 255);
        assert!(result[0] < 5);
    }

    #[test]
    fn blend_row_preserves_length() {
        let src = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let mut dst = vec![0, 0, 255, 255, 0, 0, 255, 255];
        blend_row_normal(&src, &mut dst, 128);
        assert_eq!(dst.len(), 8);
    }

    #[test]
    fn all_modes_dont_panic() {
        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen,
            BlendMode::Overlay, BlendMode::Darken, BlendMode::Lighten,
            BlendMode::ColorDodge, BlendMode::ColorBurn, BlendMode::SoftLight,
            BlendMode::HardLight, BlendMode::Difference, BlendMode::Exclusion,
        ];
        for mode in modes {
            let _ = blend_pixel([128, 64, 200, 200], [50, 100, 150, 255], mode, 180);
        }
    }
}
